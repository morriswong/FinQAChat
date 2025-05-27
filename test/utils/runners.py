"""
Test runners for different types of FinQA tests
"""

import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

from .extraction import extract_percentage_from_response, validate_percentage_answer


def run_basic_question(app, question: str, run_number: int, timeout_seconds: int = 30) -> Dict[str, Any]:
    """Run a single question and return the response with metadata"""
    print(f"\n--- Run {run_number} ---")
    print(f"Question: {question}")
    
    # Create unique session for each run
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    start_time = time.time()
    response_content = ""
    error = None
    
    try:
        print("Response: ", end="", flush=True)
        for msg, metadata in app.stream({
            "messages": [{"role": "user", "content": question}]
        }, config, stream_mode="messages"):
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                error = f"Timeout after {timeout_seconds} seconds"
                break
                
            if hasattr(msg, 'content') and msg.content:
                response_content += msg.content
                print(msg.content, end="", flush=True)
        print("\n")
        
    except Exception as e:
        error = str(e)
        print(f"\nError: {error}")
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return {
        'run_number': run_number,
        'session_id': session_id,
        'question': question,
        'response': response_content,
        'response_time_seconds': response_time,
        'timestamp': datetime.now().isoformat(),
        'error': error,
        'success': error is None
    }


def run_structured_question(app, question: str, expected_answer: str, timeout_seconds: int = 120) -> Dict[str, Any]:
    """Run a question with structured output and validation"""
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    start_time = time.time()
    response_content = ""
    error = None
    
    try:
        for msg, metadata in app.stream({
            "messages": [{"role": "user", "content": question}]
        }, config, stream_mode="messages"):
            if time.time() - start_time > timeout_seconds:
                error = f"Timeout after {timeout_seconds} seconds"
                break
                
            if hasattr(msg, 'content') and msg.content:
                response_content += msg.content
        
    except Exception as e:
        error = str(e)
    
    end_time = time.time()
    
    # Extract percentage from response
    extracted_percentage = extract_percentage_from_response(response_content)
    
    # Validate against expected answer
    answer_matches = validate_percentage_answer(extracted_percentage, expected_answer)
    
    return {
        'session_id': session_id,
        'question': question,
        'full_response': response_content,
        'response': response_content[:500] + "..." if len(response_content) > 500 else response_content,  # Truncated for summary
        'extracted_percentage': extracted_percentage,
        'expected_answer': expected_answer,
        'answer_matches': answer_matches,
        'response_time_seconds': end_time - start_time,
        'timestamp': datetime.now().isoformat(),
        'error': error,
        'success': error is None
    }


def run_performance_test(app, question: str, num_runs: int = 5, timeout_seconds: int = 60) -> list[Dict[str, Any]]:
    """Run multiple performance tests"""
    results = []
    
    for i in range(1, num_runs + 1):
        try:
            result = run_basic_question(app, question, i, timeout_seconds)
            results.append(result)            
        except KeyboardInterrupt:
            print(f"\nTest interrupted after {len(results)} runs")
            break
        except Exception as e:
            print(f"Error in run {i}: {e}")
            # Add failed result
            results.append({
                'run_number': i,
                'session_id': 'error',
                'question': question,
                'response': '',
                'response_time_seconds': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'success': False
            })
    
    return results