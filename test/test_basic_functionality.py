"""
Basic functionality tests for FinQA system
"""

import pytest
from .utils.runners import run_basic_question
from .utils.file_helpers import save_results_to_json, print_test_summary


class TestBasicFunctionality:
    """Test basic FinQA system functionality"""
    
    def test_system_initialization(self, finqa_app):
        """Test that the system initializes correctly"""
        assert finqa_app is not None, "FinQA app should be initialized"
    
    def test_single_question(self, finqa_app, sample_question):
        """Test a single question execution"""
        result = run_basic_question(finqa_app, sample_question, 1)
        
        assert result['success'], f"Test failed with error: {result['error']}"
        assert result['response'], "Response should not be empty"
        assert result['response_time_seconds'] > 0, "Response time should be positive"
    
    def test_response_timeout(self, finqa_app, sample_question):
        """Test that responses complete within reasonable time"""
        result = run_basic_question(finqa_app, sample_question, 1, timeout_seconds=60)
        
        assert result['response_time_seconds'] < 60, "Response should complete within timeout"
    
    def test_multiple_sessions(self, finqa_app, sample_question):
        """Test that multiple sessions work independently"""
        results = []
        
        for i in range(1, 4):
            result = run_basic_question(finqa_app, sample_question, i)
            results.append(result)
            assert result['success'], f"Run {i} failed with error: {result['error']}"
            assert result['response'], f"Response should not be empty for run {i}"
        
        # Verify different session IDs
        session_ids = [r['session_id'] for r in results]
        assert len(set(session_ids)) == len(session_ids), "All sessions should have unique IDs"
    
    def test_workflow_with_output_saving(self, finqa_app, sample_question, logs_dir):
        """Test complete workflow and save results to JSON for inspection"""
        # Run multiple tests and collect results
        results = []
        for i in range(1, 3):  # Run 2 tests for faster execution
            result = run_basic_question(finqa_app, sample_question, i, timeout_seconds=120)
            results.append(result)
            assert result['success'], f"Run {i} failed with error: {result['error']}"
        
        # Save results to JSON file for inspection
        output_file = f"{logs_dir}/basic_test_results.json"
        save_results_to_json(results, output_file)
        
        # Print summary for user visibility
        print_test_summary(results)
        print(f"Results saved to: {output_file}")
        
        # Verify file was created
        import os
        assert os.path.exists(output_file), "Results file should be created"