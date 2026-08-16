import os
import json
import base64

import boto3

from pypdf import PdfReader
from docx import Document
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# BEDROCK
# ============================================================

bedrock = boto3.client(
    "bedrock-runtime",
    region_name=os.getenv("REGION"),
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("AWS_SECRET_KEY")
)

MODEL_ID = os.getenv(
    "AWS_BEDROCK_MODEL_ID"
)


# ============================================================
# MAIN EXTRACTION FUNCTION
# ============================================================

def extract_text(file_path, filename):

    extension = filename.lower().split(".")[-1]

    # --------------------------------------------------------
    # PDF
    # --------------------------------------------------------

    if extension == "pdf":

        reader = PdfReader(file_path)

        text = ""

        for page_number, page in enumerate(
            reader.pages,
            start=1
        ):

            page_text = page.extract_text()

            if page_text:

                text += (
                    f"\n[PAGE {page_number}]\n"
                )

                text += page_text

        return text

    # --------------------------------------------------------
    # DOCX
    # --------------------------------------------------------

    elif extension == "docx":

        doc = Document(file_path)

        text = ""

        for paragraph in doc.paragraphs:

            if paragraph.text.strip():

                text += (
                    paragraph.text +
                    "\n"
                )

        return text

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    elif extension == "txt":

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    elif extension in [
        "png",
        "jpg",
        "jpeg"
    ]:

        with open(
            file_path,
            "rb"
        ) as f:

            image_bytes = f.read()

        return extract_image_with_claude(
            image_bytes,
            extension
        )

    else:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )


# ============================================================
# CLAUDE OCR
# ============================================================

def extract_image_with_claude(
    image_bytes,
    extension
):

    if extension in ["jpg", "jpeg"]:

        media_type = "image/jpeg"

    else:

        media_type = "image/png"

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    prompt = """
Extract all text from this document image.

Preserve the meaning and structure of the document.

Return only the extracted text.
"""

    body = {

        "anthropic_version":
            "bedrock-2023-05-31",

        "max_tokens":
            8192,

        "temperature":
            0,

        "messages": [

            {
                "role": "user",

                "content": [

                    {
                        "type": "image",

                        "source": {

                            "type": "base64",

                            "media_type":
                                media_type,

                            "data":
                                image_base64
                        }
                    },

                    {
                        "type": "text",

                        "text":
                            prompt
                    }
                ]
            }
        ]
    }

    response = bedrock.invoke_model(

        modelId=MODEL_ID,

        body=json.dumps(body)
    )

    result = json.loads(
        response["body"].read()
    )

    return result["content"][0]["text"]