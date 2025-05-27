"""
Pydantic models for structured testing
"""

from pydantic import BaseModel, Field
from typing import Optional


class StructuredFinancialAnswer(BaseModel):
    """Structured format for financial answers with percentage extraction"""
    final_answer: str = Field(description="The final numerical answer (e.g., '14.1%', '25.5%')")
    calculation_steps: list[str] = Field(description="Step-by-step calculation breakdown")
    source_values: dict = Field(description="Key financial values used in calculation")
    confidence_level: str = Field(description="Confidence in the answer: high, medium, low")


class TestResult(BaseModel):
    """Standard test result structure"""
    session_id: str
    question: str
    response: str
    response_time_seconds: float
    timestamp: str
    error: Optional[str]
    success: bool


class StructuredTestResult(TestResult):
    """Extended test result with structured validation"""
    extracted_percentage: Optional[str]
    expected_answer: str
    answer_matches: bool
    full_response: str