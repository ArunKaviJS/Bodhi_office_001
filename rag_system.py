import os
import tempfile
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from mistralai import Mistral

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI



from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# --------------------------------------------------
# Load Environment Variables
# --------------------------------------------------
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_AI_API_KEY")

AZURE_EMBEDDED_API_KEY = os.getenv("AZURE__EMBEDDED_API_KEY")
AZURE_EMBEDD_API_VERSION = os.getenv("AZURE_EMBEDD_API_VERSION")
AZURE_EMBEDD_ENDPOINT = os.getenv("AZURE_EMBEDD_ENDPOINT")
AZURE_EMBED_DEPLOYMENT_NAME = os.getenv("AZURE_EMBED_DEPLOYMENT_NAME")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT")

# --------------------------------------------------
# Initialize Clients
# --------------------------------------------------
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

embedding_model = AzureOpenAIEmbeddings(
    azure_deployment=AZURE_EMBED_DEPLOYMENT_NAME,
    azure_endpoint=AZURE_EMBEDD_ENDPOINT,
    api_key=AZURE_EMBEDDED_API_KEY,
    api_version=AZURE_EMBEDD_API_VERSION,
)

llm = AzureChatOpenAI(
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    temperature=0
)

# --------------------------------------------------
# OCR FUNCTION (MISTRAL)
# --------------------------------------------------
def extract_text_from_file(file_path):
    with open(file_path, "rb") as f:
        uploaded_file = mistral_client.files.upload(
            file={
                "file_name": os.path.basename(file_path),
                "content": f,
            },
            purpose="ocr"
        )

    response = mistral_client.ocr.process(
        model="mistral-ocr-latest",
        document={
            "type": "file",
            "file_id": uploaded_file.id,
        }
    )

    full_text = ""
    for page in response.pages:
        full_text += page.markdown + "\n"

    return full_text

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.set_page_config(page_title="📘 BODHI POC MODEL", layout="wide")

st.title("📘 BODHI POC MODEL")

uploaded_file = st.file_uploader("Upload PDF/Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    with st.spinner("🔍BODHI THINKING"):
        raw_text = extract_text_from_file(temp_path)

    st.subheader("📄 Extracted Text")
    st.text_area("OCR Output", raw_text[:2000], height=250)

    # --------------------------------------------------
    # Text Splitting
    # --------------------------------------------------
    splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(raw_text)

    documents = [Document(page_content=chunk) for chunk in chunks]

    # --------------------------------------------------
    # Create FAISS Vector Store
    # --------------------------------------------------
    with st.spinner("📊bODHI THINKING"):
        vectorstore = FAISS.from_documents(documents, embedding_model)

    st.success("✅ Document Indexed Successfully")

    # --------------------------------------------------
    # User Question
    # --------------------------------------------------
    query = st.text_input("💬 Ask a question about the document")

    if st.button("Generate Answer") and query:

        # Embed query
        query_embedding = embedding_model.embed_query(query)

        # Similarity search with scores
        results = vectorstore.similarity_search_with_score(query, k=1)

        # Filter cosine similarity >= 0.80
        filtered_docs = []
        for doc, score in results:
            similarity = 1 - score  # Convert distance to similarity
            if similarity >= 0.30:
                filtered_docs.append(doc.page_content)

        if not filtered_docs:
            st.warning("⚠️ No highly relevant context found (similarity < 80%)")
        else:
            context = "\n\n".join(filtered_docs)

            prompt = f"""
You are a helpful assistant.
Answer ONLY based on the provided document context.

Context:
{context}

Question:
{query}

Provide a clear, structured answer.
"""

            with st.spinner("🤖 BODHI THINING"):
                response = llm.invoke(prompt)

            st.subheader("📌 Final Answer")
            st.success(response.content)
