"""
File handling utilities for test results
"""

import os
import json
from typing import Dict, Any, List


def save_results_to_json(results: List[Dict[str, Any]], filename: str) -> None:
    """Save results to JSON file"""
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(results, jsonfile, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to {filename}")


def load_ground_truth_data(filepath: str) -> Dict[str, Any]:
    """Load ground truth data from train.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_ground_truth_answer(data: List[Dict], question: str) -> str:
    """Find ground truth answer for a specific question"""
    for item in data:
        if 'qa' in item and item['qa']['question'] == question:
            return item['qa']['answer']
    raise ValueError(f"Question not found in ground truth data: {question}")


def print_test_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary of test results"""
    if not results:
        print("No results to summarize.")
        return
    
    successful_runs = sum(1 for r in results if r['success'])
    
    if successful_runs > 0:
        avg_response_time = sum(r['response_time_seconds'] for r in results if r['success']) / successful_runs
    else:
        avg_response_time = 0
    
    print(f"\n=== Test Summary ===")
    print(f"Completed runs: {len(results)}")
    print(f"Successful runs: {successful_runs}/{len(results)}")
    print(f"Average response time: {avg_response_time:.2f} seconds")
    
    if successful_runs < len(results):
        failed_runs = [r for r in results if not r['success']]
        print(f"Failed runs: {len(failed_runs)}")
        for failed_run in failed_runs:
            run_num = failed_run.get('run_number', 'unknown')
            print(f"  Run {run_num}: {failed_run['error']}")


def print_structured_summary(result: Dict[str, Any]) -> None:
    """Print summary for structured test results"""
    print(f"\n=== Structured Test Results ===")
    print(f"Question: {result['question']}")
    print(f"Expected Answer: {result['expected_answer']}")
    print(f"Extracted Answer: {result['extracted_percentage']}")
    print(f"Answer Match: {result['answer_matches']}")
    print(f"Response Time: {result['response_time_seconds']:.2f} seconds")