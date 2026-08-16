import os
import json
import uuid

from pymongo import MongoClient
from openai import AzureOpenAI
from embedding import create_embedding
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# MONGODB
# ============================================================

mongo_client = MongoClient(
    os.getenv("MONGO_URI")
)

db = mongo_client["graph_rag"]


documents_collection = db["documents"]

chunks_collection = db["chunks"]

nodes_collection = db["graph_nodes"]

edges_collection = db["graph_edges"]


# ============================================================
# AZURE OPENAI
# ============================================================

llm = AzureOpenAI(

    api_key=os.getenv(
        "AZURE_OPENAI_API_KEY"
    ),

    azure_endpoint=os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    ),

    api_version=os.getenv(
        "AZURE_OPENAI_API_VERSION"
    )
)


CHAT_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_DEPLOYMENT"
)


# ============================================================
# CHUNKING
# ============================================================

def create_chunks(
    text,
    chunk_size=2000,
    overlap=300
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk_text = text[start:end]

        chunk_id = str(
            uuid.uuid4()
        )

        chunks.append({

            "chunk_id":
                chunk_id,

            "text":
                chunk_text
        })

        start = end - overlap

    return chunks


# ============================================================
# TRIPLE EXTRACTION
# ============================================================

def extract_triples(chunk_text):

    prompt = f"""
Extract knowledge graph triples from the text.

Return ONLY JSON.

Format:

{{
    "triples": [
        {{
            "head": "entity",
            "head_type": "type",
            "relation": "RELATION",
            "tail": "entity",
            "tail_type": "type",
            "confidence": 0.95
        }}
    ]
}}

Rules:

- Extract only information present in the text.
- Do not invent entities.
- Do not invent relationships.
- Use meaningful entity types.
- Use short normalized relationship names.
- Confidence must be between 0 and 1.

TEXT:

{chunk_text}
"""

    response = llm.chat.completions.create(

        model=CHAT_DEPLOYMENT,

        temperature=0,

        messages=[

            {
                "role": "system",

                "content":
                    "You extract knowledge graph triples."
            },

            {
                "role": "user",

                "content":
                    prompt
            }
        ]
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    # Remove markdown JSON wrapper

    content = (
        content
        .replace("```json", "")
        .replace("```", "")
        .strip()
    )

    return json.loads(content)


# ============================================================
# GET OR CREATE NODE
# ============================================================

def get_or_create_node(

    document_id,
    chunk_id,
    name,
    node_type
):

    name = name.strip()

    # Find existing entity

    existing = nodes_collection.find_one({

        "name": name,

        "type": node_type
    })

    if existing:

        nodes_collection.update_one(

            {
                "node_id":
                    existing["node_id"]
            },

            {
                "$addToSet": {

                    "source_chunks":
                        chunk_id,

                    "source_documents":
                        document_id
                }
            }
        )

        return existing["node_id"]


    # Create new entity

    node_id = str(
        uuid.uuid4()
    )

    node = {

        "node_id":
            node_id,

        "name":
            name,

        "type":
            node_type,

        "source_chunks":
            [chunk_id],

        "source_documents":
            [document_id]
    }

    nodes_collection.insert_one(
        node
    )

    return node_id


# ============================================================
# CREATE EDGE
# ============================================================

def create_edge(

    document_id,
    chunk_id,

    source_node,
    relation,
    target_node,

    confidence
):

    existing = edges_collection.find_one({

        "source_node":
            source_node,

        "relation":
            relation,

        "target_node":
            target_node
    })


    if existing:

        edges_collection.update_one(

            {
                "edge_id":
                    existing["edge_id"]
            },

            {
                "$addToSet": {

                    "source_chunks":
                        chunk_id,

                    "source_documents":
                        document_id
                },

                "$max": {

                    "confidence":
                        confidence
                }
            }
        )

        return


    edge_id = str(
        uuid.uuid4()
    )

    edge = {

        "edge_id":
            edge_id,

        "source_node":
            source_node,

        "relation":
            relation,

        "target_node":
            target_node,

        "confidence":
            confidence,

        "source_chunks":
            [chunk_id],

        "source_documents":
            [document_id]
    }

    edges_collection.insert_one(
        edge
    )


# ============================================================
# BUILD COMPLETE GRAPH
# ============================================================

def build_graph(

    document_id,
    filename,
    text
):

    # --------------------------------------------------------
    # Document
    # --------------------------------------------------------

    documents_collection.insert_one({

        "document_id":
            document_id,

        "filename":
            filename,

        "status":
            "PROCESSING"
    })


    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    chunks = create_chunks(text)


    # --------------------------------------------------------
    # Process each chunk
    # --------------------------------------------------------

    for index, chunk in enumerate(chunks):

        chunk_id = chunk["chunk_id"]

        chunk_text = chunk["text"]

        print(
            f"Processing chunk "
            f"{index + 1}/{len(chunks)}"
        )


        # ----------------------------------------------------
        # Create embedding
        # ----------------------------------------------------

        embedding = create_embedding(
            chunk_text
        )


        # ----------------------------------------------------
        # Store chunk + embedding
        # ----------------------------------------------------

        chunks_collection.insert_one({

            "chunk_id":
                chunk_id,

            "document_id":
                document_id,

            "chunk_index":
                index,

            "text":
                chunk_text,

            "embedding":
                embedding
        })


        # ----------------------------------------------------
        # Extract graph triples
        # ----------------------------------------------------

        result = extract_triples(
            chunk_text
        )


        for triple in result.get(
            "triples",
            []
        ):

            confidence = float(
                triple.get(
                    "confidence",
                    0
                )
            )


            # Ignore weak triples

            if confidence < 0.5:

                continue


            # ------------------------------------------------
            # Create HEAD node
            # ------------------------------------------------

            head_id = get_or_create_node(

                document_id=

                    document_id,

                chunk_id=

                    chunk_id,

                name=

                    triple["head"],

                node_type=

                    triple["head_type"]
            )


            # ------------------------------------------------
            # Create TAIL node
            # ------------------------------------------------

            tail_id = get_or_create_node(

                document_id=

                    document_id,

                chunk_id=

                    chunk_id,

                name=

                    triple["tail"],

                node_type=

                    triple["tail_type"]
            )


            # ------------------------------------------------
            # Create EDGE
            # ------------------------------------------------

            create_edge(

                document_id=

                    document_id,

                chunk_id=

                    chunk_id,

                source_node=

                    head_id,

                relation=

                    triple["relation"],

                target_node=

                    tail_id,

                confidence=

                    confidence
            )


    # --------------------------------------------------------
    # Update document
    # --------------------------------------------------------

    documents_collection.update_one(

        {
            "document_id":
                document_id
        },

        {
            "$set": {

                "status":
                    "COMPLETED",

                "chunk_count":
                    len(chunks),

                "node_count":
                    nodes_collection.count_documents({

                        "source_documents":
                            document_id
                    }),

                "edge_count":
                    edges_collection.count_documents({

                        "source_documents":
                            document_id
                    })
            }
        }
    )


    return {

        "chunks":
            len(chunks),

        "nodes":
            nodes_collection.count_documents({

                "source_documents":
                    document_id
            }),

        "edges":
            edges_collection.count_documents({

                "source_documents":
                    document_id
            })
    }