import os
import math

from pymongo import MongoClient
from openai import AzureOpenAI
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# MONGODB
# ============================================================

client = MongoClient(
    os.getenv("MONGO_URI")
)

db = client["graph_rag"]

chunks_collection = db["chunks"]

nodes_collection = db["graph_nodes"]

edges_collection = db["graph_edges"]


# ============================================================
# AZURE OPENAI - EMBEDDING CLIENT
# ============================================================

embedding_client = AzureOpenAI(

    api_key=os.getenv(
        "AZURE_OPENAI_API_KEY"
    ),

    azure_endpoint=os.getenv(
        "AZURE_OPENAI_ENDPOINT"
    ),

    api_version=os.getenv(
        "AZURE_OPENAI_EMBED_API_VERSION"
    )
)


EMBED_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBED_DEPLOYMENT"
)


# ============================================================
# AZURE OPENAI - CHAT/ANSWER CLIENT
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
# QUERY EMBEDDING
# ============================================================

def embed_query(query):

    response = embedding_client.embeddings.create(

        model=EMBED_DEPLOYMENT,

        input=query
    )

    return response.data[0].embedding


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    norm_a = math.sqrt(
        sum(x * x for x in a)
    )

    norm_b = math.sqrt(
        sum(x * x for x in b)
    )

    if norm_a == 0 or norm_b == 0:

        return 0

    return dot / (
        norm_a * norm_b
    )


# ============================================================
# VECTOR SEARCH
# ============================================================

def vector_search(
    query,
    top_k=5
):

    query_embedding = embed_query(
        query
    )

    chunks = list(
        chunks_collection.find(
            {},
            {
                "_id": 0
            }
        )
    )

    results = []

    for chunk in chunks:

        score = cosine_similarity(

            query_embedding,

            chunk["embedding"]
        )

        results.append({

            "chunk_id":
                chunk["chunk_id"],

            "document_id":
                chunk["document_id"],

            "text":
                chunk["text"],

            "score":
                score
        })


    results.sort(

        key=lambda x: x["score"],

        reverse=True
    )


    return results[:top_k]


# ============================================================
# GRAPH SEARCH
# ============================================================

def graph_search(
    chunk_ids
):

    # --------------------------------------------------------
    # Find nodes connected to retrieved chunks
    # --------------------------------------------------------

    nodes = list(
        nodes_collection.find(

            {
                "source_chunks": {
                    "$in": chunk_ids
                }
            },

            {
                "_id": 0
            }
        )
    )


    node_ids = [

        node["node_id"]

        for node in nodes
    ]


    if not node_ids:

        return {

            "nodes": [],

            "edges": []
        }


    # --------------------------------------------------------
    # Find edges connected to those nodes
    # --------------------------------------------------------

    edges = list(
        edges_collection.find(

            {
                "$or": [

                    {
                        "source_node": {
                            "$in": node_ids
                        }
                    },

                    {
                        "target_node": {
                            "$in": node_ids
                        }
                    }

                ]
            },

            {
                "_id": 0
            }
        )
    )


    return {

        "nodes":
            nodes,

        "edges":
            edges
    }


# ============================================================
# HYBRID SEARCH
# ============================================================

def hybrid_search(query):

    # --------------------------------------------------------
    # 1. Vector Search
    # --------------------------------------------------------

    vector_results = vector_search(

        query,

        top_k=5
    )


    # --------------------------------------------------------
    # 2. Get Chunk IDs
    # --------------------------------------------------------

    chunk_ids = [

        item["chunk_id"]

        for item in vector_results
    ]


    # --------------------------------------------------------
    # 3. Graph Search
    # --------------------------------------------------------

    graph_results = graph_search(

        chunk_ids
    )


    # --------------------------------------------------------
    # 4. Return combined results
    # --------------------------------------------------------

    return {

        "query":
            query,

        "vector_results":
            vector_results,

        "graph_results":
            graph_results
    }


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    retrieval_result
):

    context = ""


    # ========================================================
    # SOURCE TEXT
    # ========================================================

    context += "SOURCE DOCUMENTS:\n\n"


    for item in retrieval_result[
        "vector_results"
    ]:

        context += (

            f"Chunk ID: "
            f"{item['chunk_id']}\n"

            f"Similarity: "
            f"{item['score']:.4f}\n"

            f"Text:\n"
            f"{item['text']}\n\n"
        )


    # ========================================================
    # KNOWLEDGE GRAPH
    # ========================================================

    context += "\nKNOWLEDGE GRAPH:\n\n"


    nodes = retrieval_result[
        "graph_results"
    ]["nodes"]


    edges = retrieval_result[
        "graph_results"
    ]["edges"]


    # --------------------------------------------------------
    # Convert node ID -> node name
    # --------------------------------------------------------

    node_map = {

        node["node_id"]:
            node["name"]

        for node in nodes
    }


    # --------------------------------------------------------
    # Add graph relationships
    # --------------------------------------------------------

    for edge in edges:

        source = node_map.get(

            edge["source_node"],

            edge["source_node"]
        )


        target = node_map.get(

            edge["target_node"],

            edge["target_node"]
        )


        context += (

            f"{source} "
            f"→ {edge['relation']} → "
            f"{target}\n"
        )


    # ========================================================
    # LLM PROMPT
    # ========================================================

    prompt = f"""
You are a grounded Hybrid Graph RAG assistant.

Answer the user's question using ONLY the
provided source documents and knowledge graph.

Do not invent information.

If the answer is not available in the
provided context, say:

"I could not find the answer in the provided documents."

USER QUESTION:

{query}


CONTEXT:

{context}


Provide a clear and concise answer.
"""


    # ========================================================
    # AZURE OPENAI
    # ========================================================

    response = llm.chat.completions.create(

        model=CHAT_DEPLOYMENT,

        temperature=0,

        messages=[

            {
                "role": "system",

                "content":
                    "You are a grounded RAG assistant."
            },

            {
                "role": "user",

                "content":
                    prompt
            }
        ]
    )


    return response.choices[0].message.content