import os

from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# AZURE OPENAI EMBEDDING CLIENT
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
# CREATE EMBEDDING
# ============================================================

def create_embedding(text):

    response = embedding_client.embeddings.create(

        model=EMBED_DEPLOYMENT,

        input=text
    )

    return response.data[0].embedding