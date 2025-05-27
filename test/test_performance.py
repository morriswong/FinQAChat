"""
Performance and load tests for FinQA system
"""

import pytest
from .utils.runners import run_performance_test, run_basic_question
from .utils.file_helpers import save_results_to_json, print_test_summary


class TestPerformance:
    """Performance testing for FinQA system"""
    
    @pytest.mark.parametrize("run_number", [1, 2, 3, 4, 5])
    def test_multiple_runs(self, finqa_app, sample_question, run_number):
        """Test multiple runs of the same question"""
        result = run_basic_question(finqa_app, sample_question, run_number)
        
        assert result['success'], f"Run {run_number} failed with error: {result['error']}"
        assert result['response'], f"Response should not be empty for run {run_number}"
    
    def test_performance_benchmark(self, finqa_app, sample_question, logs_dir):
        """Run performance benchmark test"""
        print("\n=== Performance Benchmark Test ===")
        print(f"Question: {sample_question}")
        print("Running 5 iterations...")
        
        results = run_performance_test(finqa_app, sample_question, num_runs=5, timeout_seconds=60)
        
        # Basic assertions
        assert len(results) == 5, "Should complete all 5 runs"
        successful_runs = [r for r in results if r['success']]
        assert len(successful_runs) >= 3, "At least 3/5 runs should succeed"
        
        # Performance metrics
        if successful_runs:
            response_times = [r['response_time_seconds'] for r in successful_runs]
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            print(f"\nPerformance Metrics:")
            print(f"Average response time: {avg_time:.2f}s")
            print(f"Min response time: {min_time:.2f}s")
            print(f"Max response time: {max_time:.2f}s")
            print(f"Success rate: {len(successful_runs)}/5")
            
            # Performance assertions
            assert avg_time < 30, f"Average response time should be under 30s, got {avg_time:.2f}s"
            assert max_time < 60, f"Max response time should be under 60s, got {max_time:.2f}s"
        
        # Save detailed results
        output_file = f"{logs_dir}/performance_benchmark.json"
        save_results_to_json(results, output_file)
        print_test_summary(results)
        print(f"Performance results saved to: {output_file}")
    
    def test_concurrent_sessions(self, finqa_app, sample_question):
        """Test that concurrent sessions don't interfere with each other"""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def run_test(question, run_id):
            result = run_basic_question(finqa_app, question, run_id, timeout_seconds=45)
            results_queue.put(result)
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=run_test, args=(sample_question, i+1))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)  # 60 second timeout per thread
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        # Verify all threads completed successfully
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        
        for i, result in enumerate(results):
            assert result['success'], f"Concurrent run {i+1} failed: {result['error']}"
            assert result['response'], f"Concurrent run {i+1} should have response"
        
        print(f"\nConcurrent test completed successfully with {len(results)} sessions")
    
    def test_stress_test_quick(self, finqa_app, sample_question, logs_dir):
        """Quick stress test with rapid sequential requests"""
        print("\n=== Quick Stress Test ===")
        
        results = []
        for i in range(3):  # Reduced from 10 for faster testing
            result = run_basic_question(finqa_app, sample_question, i+1, timeout_seconds=30)
            results.append(result)
            
            # Don't fail entire test if one request fails
            if not result['success']:
                print(f"Request {i+1} failed: {result['error']}")
        
        successful_runs = [r for r in results if r['success']]
        success_rate = len(successful_runs) / len(results)
        
        print(f"Stress test success rate: {len(successful_runs)}/{len(results)} ({success_rate:.1%})")
        
        # Save stress test results
        output_file = f"{logs_dir}/stress_test_results.json"
        save_results_to_json(results, output_file)
        print(f"Stress test results saved to: {output_file}")
        
        # Should have at least 66% success rate under stress
        assert success_rate >= 0.66, f"Success rate under stress should be ≥66%, got {success_rate:.1%}"