# FinQA Test Suite

This directory contains the restructured test suite for the FinQA Chat system.

## Structure

```
test/
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_basic_functionality.py # Basic system functionality tests
├── test_structured_output.py   # Structured output and validation tests  
├── test_performance.py         # Performance and load tests
├── test_script.py              # Legacy test script (deprecated)
└── utils/                      # Test utility modules
    ├── __init__.py
    ├── extraction.py           # Answer extraction utilities
    ├── file_helpers.py         # File I/O and result handling
    ├── runners.py              # Test execution runners
    └── test_models.py          # Pydantic models for structured data
```

## Usage

### Run All Tests
```bash
pytest test/ -v
```

### Run Specific Test Categories

#### Basic Functionality Tests
```bash
pytest test/test_basic_functionality.py -v -s
```

#### Structured Output Tests (with ground truth validation)
```bash
pytest test/test_structured_output.py -v -s
```

#### Performance Tests
```bash
pytest test/test_performance.py -v -s
```

### Run Individual Tests

#### Single question test
```bash
pytest test/test_basic_functionality.py::TestBasicFunctionality::test_single_question -v -s
```

#### Structured validation test
```bash
pytest test/test_structured_output.py::TestStructuredOutput::test_structured_output_with_validation -v -s
```

#### Performance benchmark
```bash
pytest test/test_performance.py::TestPerformance::test_performance_benchmark -v -s
```

#### Sample evaluation with ground truth
```bash
pytest test/test_evaluation.py::TestEvaluation::test_sample_evaluation_with_ground_truth -v -s
```

## Test Categories

### 1. Basic Functionality Tests (`test_basic_functionality.py`)
- System initialization
- Single question execution
- Response timeout handling
- Multiple session independence
- Basic output file generation

### 2. Structured Output Tests (`test_structured_output.py`)
- Percentage extraction from responses
- Ground truth validation (14.1% from train.json)
- Answer accuracy with tolerance levels
- Consistency across multiple runs
- Structured result analysis

### 3. Performance Tests (`test_performance.py`)
- Parametrized multiple runs
- Performance benchmarking
- Concurrent session testing
- Stress testing with rapid requests

### 4. Evaluation Tests (`test_evaluation.py`)
- Sample dataset evaluation with ground truth comparison
- Accuracy validation against ConvFinQA data
- Side-by-side result comparison
- Automated accuracy metrics calculation

## Output Files

Test results are saved to `./logs/` directory:
- `basic_test_results.json` - Basic functionality test results
- `structured_test_results.json` - Structured validation results
- `accuracy_analysis.json` - Answer accuracy analysis
- `consistency_analysis.json` - Multi-run consistency analysis
- `performance_benchmark.json` - Performance metrics
- `stress_test_results.json` - Stress test results
- `evaluation_sample_results.json` - Sample dataset evaluation results

## Key Features

### Ground Truth Validation
The structured tests validate against the ground truth answer from `data/train.json`:
- Question: "what was the percentage change in the net cash from operating activities from 2008 to 2009"
- Expected Answer: "14.1%"

### Robust Answer Extraction
Multiple regex patterns extract percentages from various response formats:
- "14.1%"
- "14.1 percent" 
- "is 14.1%"
- "answer: 14.1%"

### Configurable Tolerance
Answer validation supports configurable tolerance levels for floating-point comparison.

### Comprehensive Logging
All tests generate detailed JSON logs for debugging and analysis.

## Fixtures

### `finqa_app` (session-scoped)
Initializes the complete FinQA system once per test session.

### `sample_question`
Provides the standard test question.

### `expected_answer`
Provides the ground truth answer (14.1%).

### `logs_dir`
Ensures the logs directory exists for output files.