#!/usr/bin/env python3
"""
Pytest test suite for the FinQA chat system
"""

import sys
import os
import json
import time
import uuid
import pytest
import re
from datetime import datetime
from pydantic import BaseModel, Field
from typing import Optional

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

# Import from src module
from src.config import get_llm_config, DATASET_PATH
from src.finqa_rag import FinancialRAGSystem  
from src.tools import create_financial_context_lookup_tool, calculator
from src.agents import create_math_agent, create_financial_research_agent
from src.workflow import create_application_workflow
from langchain_openai import ChatOpenAI

class StructuredFinancialAnswer(BaseModel):
    """Structured format for financial answers with percentage extraction"""
    final_answer: str = Field(description="The final numerical answer (e.g., '14.1%', '25.5%')")
    calculation_steps: list[str] = Field(description="Step-by-step calculation breakdown")
    source_values: dict = Field(description="Key financial values used in calculation")
    confidence_level: str = Field(description="Confidence in the answer: high, medium, low")

def extract_percentage_from_response(response: str) -> Optional[str]:
    """Extract percentage value from response text"""
    # Look for percentage patterns like "14.1%" or "14.1 percent"
    percentage_patterns = [
        r'(\d+\.?\d*)\s*%',  # Matches "14.1%" or "14%"
        r'(\d+\.?\d*)\s*percent',  # Matches "14.1 percent"
        r'is\s+(\d+\.?\d*)\s*%',  # Matches "is 14.1%"
        r'answer:\s*(\d+\.?\d*)\s*%',  # Matches "answer: 14.1%"
    ]
    
    for pattern in percentage_patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return f"{match.group(1)}%"
    
    return None

def run_structured_question(app, question, expected_answer, timeout_seconds=120):
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
    answer_matches = False
    if extracted_percentage and expected_answer:
        # Normalize both answers for comparison
        extracted_num = float(extracted_percentage.replace('%', ''))
        expected_num = float(expected_answer.replace('%', ''))
        # Allow small tolerance for floating point comparison
        answer_matches = abs(extracted_num - expected_num) < 0.1
    
    return {
        'session_id': session_id,
        'question': question,
        'full_response': response_content,
        'extracted_percentage': extracted_percentage,
        'expected_answer': expected_answer,
        'answer_matches': answer_matches,
        'response_time_seconds': end_time - start_time,
        'timestamp': datetime.now().isoformat(),
        'error': error,
        'success': error is None
    }

@pytest.fixture(scope="session")
def finqa_app():
    """Initialize the FinQA system components (session-scoped fixture)"""
    print("Initializing Financial Chat System...")
    
    # 1. Initialize LLM
    llm_configs = get_llm_config()
    llm = ChatOpenAI(**llm_configs)

    # 2. Initialize RAG System
    rag_system_instance = FinancialRAGSystem(dataset_path=DATASET_PATH)

    # 3. Create Tools
    financial_lookup_tool = create_financial_context_lookup_tool(rag_system_instance)

    # 4. Create Agents
    math_agent_instance = create_math_agent(llm)
    financial_agent_instance = create_financial_research_agent(llm, financial_lookup_tool)

    # 5. Create Workflow
    app = create_application_workflow(llm, math_agent_instance, financial_agent_instance)
    
    return app

def run_question(app, question, run_number, timeout_seconds=30):
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
        message_count = 0
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
                message_count += 1
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

def save_results_to_json(results, filename='test_results.json'):
    """Save results to JSON file"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4)
    
    print(f"\nResults saved to {filename}")

class TestFinQASystem:
    """Test class for FinQA chat system"""
    
    def test_single_question(self, finqa_app):
        """Test a single question execution"""
        question = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
        result = run_question(finqa_app, question, 1)
        
        assert result['success'], f"Test failed with error: {result['error']}"
        assert result['response'], "Response should not be empty"
        assert result['response_time_seconds'] > 0, "Response time should be positive"
    
    def test_complete_workflow_with_output(self, finqa_app):
        """Test complete workflow and save results to JSON for inspection"""
        question = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
        
        # Run multiple tests and collect results
        results = []
        for i in range(1, 2):
            result = run_question(finqa_app, question, i, timeout_seconds=120)  # Increased timeout for complete responses
            results.append(result)
            assert result['success'], f"Run {i} failed with error: {result['error']}"
        
        # Save results to JSON file for inspection
        os.makedirs('./logs', exist_ok=True)
        save_results_to_json(results, './logs/pytest_test_results.json')
        
        # Verify file was created
        assert os.path.exists('./logs/pytest_test_results.json'), "Results file should be created"
        
        # Print summary for user visibility
        successful_runs = sum(1 for r in results if r['success'])
        avg_response_time = sum(r['response_time_seconds'] for r in results) / len(results)
        
        print(f"\n=== Test Results Summary ===")
        print(f"Completed runs: {len(results)}")
        print(f"Successful runs: {successful_runs}/{len(results)}")
        print(f"Average response time: {avg_response_time:.2f} seconds")
        print(f"Results saved to: ./logs/pytest_test_results.json")
    
    def test_structured_output_with_validation(self, finqa_app):
        """Test with structured output extraction and ground truth validation"""
        question = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
        expected_answer = "14.1%"  # Ground truth from train.json
        
        result = run_structured_question(finqa_app, question, expected_answer)
        
        # Basic success assertion
        assert result['success'], f"Test failed with error: {result['error']}"
        
        # Response should not be empty
        assert result['full_response'], "Response should not be empty"
        
        # Should extract a percentage
        assert result['extracted_percentage'], f"No percentage found in response. Response: {result['full_response'][:200]}..."
        
        # Should match expected answer
        assert result['answer_matches'], f"Expected {expected_answer}, got {result['extracted_percentage']}"
        
        # Save detailed results
        os.makedirs('./logs', exist_ok=True)
        save_results_to_json([result], './logs/structured_test_results.json')
        
        print(f"\n=== Structured Test Results ===")
        print(f"Question: {question}")
        print(f"Expected Answer: {expected_answer}")
        print(f"Extracted Answer: {result['extracted_percentage']}")
        print(f"Answer Match: {result['answer_matches']}")
        print(f"Response Time: {result['response_time_seconds']:.2f} seconds")
        print(f"Detailed results saved to: ./logs/structured_test_results.json")
    
    @pytest.mark.parametrize("run_number", [1, 2, 3, 4, 5])
    def test_multiple_runs(self, finqa_app, run_number):
        """Test multiple runs of the same question"""
        question = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
        result = run_question(finqa_app, question, run_number)
        
        assert result['success'], f"Run {run_number} failed with error: {result['error']}"
        assert result['response'], f"Response should not be empty for run {run_number}"
    
    def test_system_initialization(self, finqa_app):
        """Test that the system initializes correctly"""
        assert finqa_app is not None, "FinQA app should be initialized"
    
    def test_response_timeout(self, finqa_app):
        """Test that responses complete within reasonable time"""
        question = "what was the percentage change in the net cash from operating activities from 2008 to 2009"
        result = run_question(finqa_app, question, 1, timeout_seconds=60)
        
        assert result['response_time_seconds'] < 60, "Response should complete within timeout"

def test_save_results_functionality(tmp_path):
    """Test the result saving functionality"""
    test_results = [
        {
            'run_number': 1,
            'session_id': 'test-123',
            'question': 'test question',
            'response': 'test response',
            'response_time_seconds': 1.5,
            'timestamp': datetime.now().isoformat(),
            'error': None,
            'success': True
        }
    ]
    
    test_file = tmp_path / "test_results.json"
    save_results_to_json(test_results, str(test_file))
    
    assert test_file.exists(), "Results file should be created"
    
    with open(test_file, 'r') as f:
        loaded_results = json.load(f)
    
    assert len(loaded_results) == 1, "Should save one result"
    assert loaded_results[0]['success'] == True, "Should preserve success status"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])