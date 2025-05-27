"""
Utilities for extracting and validating answers from responses
"""

import re
from typing import Optional


def extract_percentage_from_response(response: str) -> Optional[str]:
    """Extract percentage value from response text"""
    # Look for percentage patterns like "14.1%" or "14.1 percent"
    percentage_patterns = [
        r'(\d+\.?\d*)\s*%',  # Matches "14.1%" or "14%"
        r'(\d+\.?\d*)\s*percent',  # Matches "14.1 percent"
        r'is\s+(\d+\.?\d*)\s*%',  # Matches "is 14.1%"
        r'answer:\s*(\d+\.?\d*)\s*%',  # Matches "answer: 14.1%"
        r'result:\s*(\d+\.?\d*)\s*%',  # Matches "result: 14.1%"
        r'(\d+\.?\d*)\s*%\s*change',  # Matches "14.1% change"
    ]
    
    for pattern in percentage_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return f"{match.group(1)}%"
    
    return None


def validate_percentage_answer(extracted: str, expected: str, tolerance: float = 0.1) -> bool:
    """Validate extracted percentage against expected answer with tolerance"""
    if not extracted or not expected:
        return False
    
    try:
        # Normalize both answers for comparison
        extracted_num = float(extracted.replace('%', ''))
        expected_num = float(expected.replace('%', ''))
        # Allow small tolerance for floating point comparison
        return abs(extracted_num - expected_num) <= tolerance
    except (ValueError, AttributeError):
        return False


def extract_numerical_value(response: str) -> Optional[float]:
    """Extract numerical value from response (without percentage sign)"""
    # Look for numerical patterns
    patterns = [
        r'(\d+\.?\d*)',  # Basic number pattern
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None