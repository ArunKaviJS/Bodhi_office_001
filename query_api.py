from fastapi import APIRouter
from pydantic import BaseModel

from retrieval import hybrid_search
from retrieval import generate_answer


router = APIRouter()


class QueryRequest(BaseModel):

    question: str


@router.post("/query")
def ask_question(
    request: QueryRequest
):

    question = request.question

    # -----------------------------
    # Hybrid retrieval
    # -----------------------------

    retrieval_result = hybrid_search(
        question
    )

    # -----------------------------
    # Generate answer
    # -----------------------------

    answer = generate_answer(
        question,
        retrieval_result
    )

    return {

        "question":
            question,

        "answer":
            answer,

        "vector_results":
            retrieval_result[
                "vector_results"
            ],

        "graph_results":
            retrieval_result[
                "graph_results"
            ]
    }