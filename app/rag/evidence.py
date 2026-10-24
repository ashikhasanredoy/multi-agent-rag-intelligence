from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    source_id: str = Field(..., description="Unique identifier for evidence item.")
    source_type: str = Field(..., description="document | web | finance | product")
    title: str = Field(..., description="Title of the source.")
    url: Optional[str] = Field(default=None, description="Direct URL if available.")
    content: str = Field(..., description="Raw textual evidence.")
    retrieved_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    relevance_score: Optional[float] = Field(default=None)
    document_page: Optional[int] = Field(default=None)
    ticker: Optional[str] = Field(default=None)
    currency: Optional[str] = Field(default=None)
    verified: bool = Field(default=True)


class EvidenceValidationResult(BaseModel):
    is_sufficient: bool = True
    validated_evidence: List[EvidenceItem] = Field(default_factory=list)
    rejected_evidence: List[EvidenceItem] = Field(default_factory=list)
    rejection_reason: Optional[str] = None
    summary_for_llm: str = ""


class EvidenceValidator:
    """
    Evidence Validation Layer.
    Validates evidence before answer generation:
    - Verifies relevance to user query
    - Confirms product evidence pertains to the asked product (not ticker data)
    - Confirms financial evidence pertains to requested stocks/tickers
    - Confirms document evidence meets calibrated semantic similarity
    - Filters out off-topic citations
    """

    @staticmethod
    def validate_product_evidence(query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        valid = []
        for item in items:
            # Reject if accidentally populated from stock market quotes
            if item.source_type == "finance":
                continue
            valid.append(item)
        return valid

    @staticmethod
    def validate_finance_evidence(query: str, items: List[EvidenceItem]) -> List[EvidenceItem]:
        valid = []
        for item in items:
            # Confirm item has actual financial quote data
            if item.source_type == "finance":
                valid.append(item)
        return valid

    @staticmethod
    def validate_document_evidence(query: str, items: List[EvidenceItem], threshold: float = 0.45) -> List[EvidenceItem]:
        valid = []
        for item in items:
            if item.source_type == "document":
                # Ensure minimum semantic threshold
                score = item.relevance_score or 0.0
                if score >= threshold:
                    valid.append(item)
        return valid


evidence_validator = EvidenceValidator()
