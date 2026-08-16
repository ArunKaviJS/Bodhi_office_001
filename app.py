import os
import uuid

from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv

from extraction import extract_text
from graph_builder import build_graph
from query_api import router


load_dotenv()

app = FastAPI(
    title="Agentic Graph Construction API"
)

app.include_router(router)


# ============================================================
# UPLOAD + GRAPH CONSTRUCTION
# ============================================================

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):

    document_id = str(uuid.uuid4())

    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join(
        "uploads",
        f"{document_id}_{file.filename}"
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    print(f"File uploaded: {file.filename}")

    # --------------------------------------------------------
    # STEP 1
    # Extract text
    # Claude Bedrock is used for image OCR
    # --------------------------------------------------------

    text = extract_text(
        file_path,
        file.filename
    )

    print("Text extraction completed.")

    # --------------------------------------------------------
    # STEP 2
    # Build chunks + embeddings + graph
    # --------------------------------------------------------

    result = build_graph(
        document_id=document_id,
        filename=file.filename,
        text=text
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "status": "success",
        "document_id": document_id,
        "filename": file.filename,
        "chunks": result["chunks"],
        "nodes": result["nodes"],
        "edges": result["edges"],
        "message": "Graph construction completed"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {
        "status": "running",
        "service": "Agentic Graph Construction"
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )