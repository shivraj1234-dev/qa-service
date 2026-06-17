"""Evaluation endpoint using LLM-as-judge scoring."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from qa_service.dependencies import get_judge_service, get_rag_service
from qa_service.models.evaluate import (
    EvalRequest,
    EvalResponse,
    EvalResultItem,
)
from qa_service.observability.logging import get_logger
from qa_service.services.judge import JudgeService
from qa_service.services.rag import RagService

router = APIRouter(tags=["evaluate"])
logger = get_logger(__name__)


@router.post("/evaluate", response_model=EvalResponse)
def evaluate(
    payload: EvalRequest,
    rag: RagService = Depends(get_rag_service),
    judge: JudgeService = Depends(get_judge_service),
) -> EvalResponse:
    """Run each test case against the RAG pipeline and score the answers."""
    results: list[EvalResultItem] = []
    for case in payload.test_cases:
        rag_response = rag.query(question=case.question, doc_id=case.doc_id)
        verdict = judge.score(
            question=case.question,
            expected=case.expected_answer,
            generated=rag_response.answer,
        )
        results.append(
            EvalResultItem(
                question=case.question,
                generated_answer=rag_response.answer,
                score=verdict.score,
                reasoning=verdict.reasoning,
            )
        )

    total = len(results)
    avg_score = sum(item.score for item in results) / total if total else 0.0
    logger.info("evaluation_complete", total=total, avg_score=avg_score)
    return EvalResponse(total=total, avg_score=avg_score, results=results)
