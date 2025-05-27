"""
Structured output tests with answer validation
"""

import pytest
from .utils.runners import run_structured_question
from .utils.file_helpers import save_results_to_json, print_structured_summary
from .utils.extraction import extract_percentage_from_response, validate_percentage_answer


class TestStructuredOutput:
    """Test structured output extraction and validation"""
    
    def test_percentage_extraction(self):
        """Test percentage extraction from various response formats"""
        test_cases = [
            ("The answer is 14.1%", "14.1%"),
            ("Result: 25 percent", "25%"),
            ("The percentage change is 14.1%", "14.1%"),
            ("Answer: 14.1%", "14.1%"),
            ("No percentage here", None),
        ]
        
        for response, expected in test_cases:
            result = extract_percentage_from_response(response)
            assert result == expected, f"Expected {expected}, got {result} for '{response}'"
    
    def test_percentage_validation(self):
        """Test percentage validation logic"""
        assert validate_percentage_answer("14.1%", "14.1%") == True
        assert validate_percentage_answer("14.0%", "14.1%") == True  # Within tolerance
        assert validate_percentage_answer("15.0%", "14.1%") == False  # Outside tolerance
        assert validate_percentage_answer(None, "14.1%") == False
        assert validate_percentage_answer("14.1%", None) == False
    
    def test_structured_output_with_validation(self, finqa_app, sample_question, expected_answer, logs_dir):
        """Test with structured output extraction and ground truth validation"""
        result = run_structured_question(finqa_app, sample_question, expected_answer)
        
        # Basic success assertion
        assert result['success'], f"Test failed with error: {result['error']}"
        
        # Response should not be empty
        assert result['full_response'], "Response should not be empty"
        
        # Should extract a percentage
        assert result['extracted_percentage'], f"No percentage found in response. Response: {result['full_response'][:200]}..."
        
        # Save detailed results first for debugging
        output_file = f"{logs_dir}/structured_test_results.json"
        save_results_to_json([result], output_file)
        
        # Print summary for debugging
        print_structured_summary(result)
        print(f"Detailed results saved to: {output_file}")
        
        # Print debug info
        print(f"\n=== Debug Information ===")
        print(f"Full response preview: {result['full_response'][:500]}...")
        
        # Should match expected answer (but don't fail the test, just warn)
        if not result['answer_matches']:
            print(f"\n⚠️  WARNING: Answer mismatch!")
            print(f"Expected: {expected_answer}")
            print(f"Got: {result['extracted_percentage']}")
            print(f"This suggests the RAG system may not be retrieving the correct data.")
            
            # Still assert for now to see the failure
            assert result['answer_matches'], f"Expected {expected_answer}, got {result['extracted_percentage']}"
    
    def test_answer_accuracy_tolerance(self, finqa_app, sample_question, logs_dir):
        """Test answer accuracy with different tolerance levels"""
        expected_answer = "14.1%"
        
        result = run_structured_question(finqa_app, sample_question, expected_answer)
        
        if result['extracted_percentage']:
            # Test with strict tolerance
            strict_match = validate_percentage_answer(
                result['extracted_percentage'], 
                expected_answer, 
                tolerance=0.01
            )
            
            # Test with loose tolerance
            loose_match = validate_percentage_answer(
                result['extracted_percentage'], 
                expected_answer, 
                tolerance=1.0
            )
            
            print(f"\nAccuracy Analysis:")
            print(f"Extracted: {result['extracted_percentage']}")
            print(f"Expected: {expected_answer}")
            print(f"Strict match (±0.01%): {strict_match}")
            print(f"Loose match (±1.0%): {loose_match}")
            
            # Save analysis
            analysis = {
                **result,
                'strict_tolerance_match': strict_match,
                'loose_tolerance_match': loose_match
            }
            
            output_file = f"{logs_dir}/accuracy_analysis.json"
            save_results_to_json([analysis], output_file)
            print(f"Analysis saved to: {output_file}")
    
    def test_multiple_structured_runs(self, finqa_app, sample_question, expected_answer, logs_dir):
        """Test consistency across multiple structured runs"""
        results = []
        
        for i in range(3):  # 3 runs for consistency testing
            result = run_structured_question(finqa_app, sample_question, expected_answer)
            results.append(result)
            
            assert result['success'], f"Run {i+1} failed: {result['error']}"
        
        # Analyze consistency
        extracted_answers = [r['extracted_percentage'] for r in results if r['extracted_percentage']]
        match_results = [r['answer_matches'] for r in results]
        
        print(f"\nConsistency Analysis:")
        print(f"Extracted answers: {extracted_answers}")
        print(f"Match results: {match_results}")
        print(f"Consistency rate: {sum(match_results)}/{len(match_results)}")
        
        # Save consistency analysis
        consistency_data = {
            'individual_results': results,
            'extracted_answers': extracted_answers,
            'match_results': match_results,
            'consistency_rate': sum(match_results) / len(match_results)
        }
        
        output_file = f"{logs_dir}/consistency_analysis.json"
        save_results_to_json([consistency_data], output_file)
        print(f"Consistency analysis saved to: {output_file}")
        
        # Assert at least 2/3 runs should match
        assert sum(match_results) >= 2, f"At least 2/3 runs should match expected answer. Got {sum(match_results)}/{len(match_results)}"