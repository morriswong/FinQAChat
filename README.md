# FinQAChat: Conversational Financial Document Analysis Agent

A sophisticated LLM-powered agent system for answering conversational questions about financial documents. This implementation goes beyond simple RAG to provide accurate financial calculations with step-by-step reasoning.

[![CI/CD Pipeline](https://github.com/morriswong/FinQAChat/actions/workflows/ci.yml/badge.svg)](https://github.com/morriswong/FinQAChat/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Overview

FinQAChat is an intelligent conversational agent that can analyze financial documents and answer complex questions requiring multi-step reasoning. The system combines retrieval-augmented generation with specialized mathematical reasoning to provide accurate percentage calculations and financial analysis.

**Example Query:**
> "What was the percentage change in the net cash from operating activities from 2008 to 2009?"

**System Response:** `14.1%` (with step-by-step calculation reasoning)

## 🏗️ Architecture

For a more detailed explanation of the architecture and design decisions, refer to the [METHODOLOGY.md](METHODOLOGY.md) document.

### High-Level Design

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Query    │───▶│   Supervisor     │───▶│   Response      │
└─────────────────┘    │   Agent          │    └─────────────────┘
                       └─────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
               ┌────▼─────┐              ┌────▼─────┐
               │Financial │              │   Math   │
               │Research  │              │  Expert  │
               │ Agent    │              │  Agent   │
               └────┬─────┘              └────┬─────┘
                    │                         │
           ┌────────▼────────┐       ┌────────▼────────┐
           │Financial Context│       │   Calculator    │
           │Lookup Tool      │       │     Tool        │
           └─────────────────┘       └─────────────────┘
```

### Core Components

1. **Supervisor Agent**: Routes queries to appropriate specialized agents using LangGraph
2. **Financial Research Agent**: Handles document lookup, context extraction, and financial data analysis
3. **Math Expert Agent**: Performs precise calculations using a secure calculator tool
4. **RAG System**: Similarity-based retrieval from financial dataset with context extraction
5. **Memory Management**: Maintains conversation history across multiple turns

### Key Features

- **Multi-turn Conversation Memory**: Maintains context across conversation turns
- **Calculation Transparency**: Shows step-by-step reasoning for complex computations
- **Numerical Precision**: Handles financial calculations with appropriate decimal precision
- **Error Handling**: Gracefully handles missing data, invalid calculations, or ambiguous references
- **Citation/Source Tracking**: Provides references to specific document sections/tables used in answers

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Local LLM server (LM Studio recommended) or OpenAI API access

### Installation

1. **Clone the repository**
   ```bash
   git clone git@github.com:morriswong/FinQAChat.git
   cd FinQAChat
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

   Required environment variables:
   ```env
   # LLM Configuration
   OPENAI_MODEL_NAME=qwen3-4b-mlx
   OPENAI_BASE_URL=http://localhost:1234/v1
   OPENAI_API_KEY=lm-studio
   MODEL_TEMPERATURE=0.1
   
   # LangSmith Tracing (optional)
   LANGCHAIN_TRACING_V2=true
   LANGSMITH_API_KEY=your_langsmith_key
   LANGSMITH_PROJECT=finqa-chat
   
   # Data Configuration
   DATASET_PATH=./data/train.json
   ```

5. **Run the chat interface**
   ```bash
   python -m src.main
   ```

## 🧪 Testing

### Run All Tests
```bash
pytest test/ -v
```

### Run Specific Test Categories

**Basic Functionality Tests**
```bash
pytest test/test_basic_functionality.py -v -s
```

**Structured Output Tests (with ground truth validation)**
```bash
pytest test/test_structured_output.py -v -s
```

**Performance Tests**
```bash
pytest test/test_performance.py -v -s
```

**Dataset Evaluation Tests (sample questions with ground truth comparison)**
```bash
pytest test/test_evaluation.py -v -s
```

### Test Results

Our test suite validates against ConvFinQA ground truth data:
- **Question**: "what was the percentage change in the net cash from operating activities from 2008 to 2009"
- **Expected Answer**: 14.1%
- **Current Status**: Under development - system shows correct reasoning but data retrieval consistency needs improvement

## 📊 Evaluation & Performance

For a comprehensive breakdown of our evaluation methodology, detailed results, and error analysis, please refer to the [METHODOLOGY.md](METHODOLOGY.md) document.

### Current Performance

| Metric | Current Status | Target | Notes |
|--------|----------------|---------|-------|
| Data Retrieval | Inconsistent | 95%+ | Sometimes finds correct data, sometimes uses fallback |
| Calculation Logic | ✅ Working | ✅ | Mathematical reasoning and formulas are correct |
| Response Time | ~35s | <30s | Average response time for complex queries |
| System Reliability | ✅ 100% | 95%+ | No crashes or errors in test suite |

### Sample Results

**Query**: "what was the percentage change in the net cash from operating activities from 2008 to 2009"

**Expected Response**:
```
Based on the financial data retrieved:
- Net cash from operating activities 2008: $181,001
- Net cash from operating activities 2009: $206,588

Calculation: ((206,588 - 181,001) / 181,001) * 100 = 14.1%

The percentage change in net cash from operating activities from 2008 to 2009 was 14.1%.
```

**Current Issues**: 
- Data retrieval sometimes uses incorrect or fallback values
- Working on improving consistency of financial context lookup

## 🏃‍♂️ Usage Examples

### Interactive Chat
```python
from src.main import run_chat_interface
run_chat_interface()
```

### Programmatic Usage
```python
from src.config import get_llm_config, DATASET_PATH
from src.finqa_rag import FinancialRAGSystem
from src.workflow import create_application_workflow
from langchain_openai import ChatOpenAI

# Initialize system
llm = ChatOpenAI(**get_llm_config())
rag_system = FinancialRAGSystem(dataset_path=DATASET_PATH)
app = create_application_workflow(llm, math_agent, financial_agent)

# Process query
result = app.invoke({
    "messages": [{"role": "user", "content": "your financial question"}]
})
```

## 📄 Detailed Methodology

For an in-depth understanding of FinQAChat's technical approach, design decisions, and evaluation methodology, please refer to our [METHODOLOGY.md](METHODOLOGY.md) document. It covers:

- Multi-Agent Architecture and its benefits
- Detailed Retrieval Strategy (Similarity-Based Matching)
- Secure Calculator Tool and Calculation Pipeline
- Memory Management using LangGraph checkpointer
- Comprehensive Evaluation Methodology and Metrics
- In-depth Results, Error Analysis, and Design Trade-offs
- Future Enhancements and Limitations

## 🛠️ Development

### Code Structure
```
src/
├── main.py           # Main entry point and chat interface
├── config.py         # Configuration management
├── workflow.py       # LangGraph supervisor workflow
├── agents.py         # Specialized agent definitions
├── tools.py          # Calculator and lookup tools
├── finqa_rag.py      # RAG system for financial document retrieval
└── archive/          # Legacy implementations and examples

test/
├── test_basic_functionality.py    # Core system tests
├── test_structured_output.py      # Ground truth validation
├── test_performance.py            # Performance benchmarks
└── utils/                         # Test utilities
```

### Key Design Decisions

1. **Multi-Agent Architecture**: Separates financial research from mathematical computation for better accuracy and maintainability
2. **LangGraph Supervisor**: Intelligent routing based on query type rather than simple RAG
3. **Similarity-Based Retrieval**: Uses difflib for robust query matching instead of embedding-based search
4. **Secure Calculator**: Sandboxed evaluation environment for mathematical expressions
5. **Memory Management**: Session-based conversation tracking with LangGraph checkpointer

## 🔮 Future Improvements

### Technical Enhancements
- [ ] **Embedding-based RAG**: Replace similarity matching with semantic embeddings
- [ ] **Multiple Model Support**: Add support for different LLM providers and local models
- [ ] **Caching Layer**: Implement response caching for improved performance
- [ ] **Advanced Parsing**: Better table and document structure understanding
- [ ] **Error Recovery**: More sophisticated error handling and retry mechanisms

### Feature Additions
- [ ] **Chart Generation**: Visual representation of financial data and trends
- [ ] **Export Capabilities**: PDF/Excel report generation
- [ ] **Audit Trail**: Detailed logging of calculation steps for compliance
- [ ] **Batch Processing**: Support for processing multiple documents
- [ ] **API Interface**: REST API for integration with other systems

### Scale & Production
- [ ] **Containerization**: Docker deployment for consistent environments
- [ ] **Monitoring**: Comprehensive observability and alerting
- [ ] **Rate Limiting**: API rate limiting and quota management
- [ ] **Data Pipeline**: Automated dataset updates and retraining
- [ ] **Security Hardening**: Enhanced input validation and security measures

## 🤝 Strengths & Limitations

### ✅ Strengths
- **Beyond Simple RAG**: Implements sophisticated multi-agent reasoning
- **Solid Architecture**: Multi-agent system with specialized financial and math experts
- **Modular Design**: Clean separation of concerns with testable components
- **Professional Quality**: Comprehensive testing, CI/CD, and documentation
- **Conversation Memory**: Maintains context across multiple turns
- **Transparent Reasoning**: Shows step-by-step calculation process
- **Correct Mathematical Logic**: Percentage calculations and formulas are accurate

### ⚠️ Current Limitations
- **Data Retrieval Consistency**: Financial context lookup needs improvement for reliable accuracy
- **Model Output Formatting**: Some models produce reasoning tokens that need filtering
- **Dataset Dependency**: Limited to ConvFinQA training data scope
- **Simple Similarity Matching**: Could benefit from semantic embedding search
- **Response Time**: Currently 35s average, target <30s
- **Single-Language Support**: Currently English-only implementation

## 📄 License

MIT License - feel free to use and modify for your projects.

## 🙏 Acknowledgments

- LangChain/LangGraph communities for the agent framework
- LM Studio for local LLM deployment capabilities
- ConvFinQA dataset for training and evaluation data
