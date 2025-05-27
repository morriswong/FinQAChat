"""
Pytest configuration and shared fixtures
"""

import sys
import os
import pytest

# Add project root to path for module imports
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.config import get_llm_config, DATASET_PATH
from src.finqa_rag import FinancialRAGSystem  
from src.tools import create_financial_context_lookup_tool
from src.agents import create_math_agent, create_financial_research_agent
from src.workflow import create_application_workflow
from langchain_openai import ChatOpenAI


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


@pytest.fixture
def sample_question():
    """Sample question for testing"""
    return "what was the percentage change in the net cash from operating activities from 2008 to 2009"


@pytest.fixture
def expected_answer():
    """Expected answer from ground truth data"""
    return "14.1%"


@pytest.fixture
def logs_dir():
    """Ensure logs directory exists"""
    logs_path = os.path.join(os.path.dirname(__file__), '..', 'logs')
    os.makedirs(logs_path, exist_ok=True)
    return logs_path