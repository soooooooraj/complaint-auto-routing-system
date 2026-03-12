from pydantic import BaseModel, Field
from typing import List, Optional

class ComplaintTextInput(BaseModel):
    text: str = Field(..., description="The complaint text to process")
    language: Optional[str] = Field(None, description="Optional language hint (e.g., 'hi', 'ml', 'mr')")

class OfficerResult(BaseModel):
    officer_id: str
    name: str
    department: str
    final_score: float
    current_workload: int

class SimilarComplaint(BaseModel):
    complaint_id: str
    text: str
    category: str
    priority: str
    similarity_score: float

class ComplaintResponse(BaseModel):
    complaint_text: str
    detected_language: str
    priority: str
    priority_confidence: float
    eta_days: int
    assigned_officers: List[OfficerResult]
    similar_complaints: List[SimilarComplaint]
    processing_time_ms: float
