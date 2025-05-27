# src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith Tracing
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_API_KEY = os.getenv('LANGSMITH_API_KEY')
LANGCHAIN_PROJECT = os.getenv('LANGSMITH_PROJECT')

# LLM Configuration
OPENAI_MODEL_NAME = os.getenv('OPENAI_MODEL_NAME', 'qwen3-4b-mlx')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', "http://localhost:1234/v1")
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', "lm-studio")
MODEL_TEMPERATURE = float(os.getenv('MODEL_TEMPERATURE', 0.1))
MODEL_STREAMING = os.getenv('MODEL_STREAMING', "True").lower() == "true"

# Data Configuration
DATASET_PATH = os.getenv('DATASET_PATH', "./data/train.json") # Relative to project root

# Ensure LangSmith vars are set if tracing is enabled
if LANGCHAIN_TRACING_V2 == "true":
    os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2
    if LANGCHAIN_API_KEY:
        os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    else:
        print("Warning: LANGSMITH_API_KEY not set, but tracing is enabled.")
    if LANGCHAIN_PROJECT:
        os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
    else:
        print("Warning: LANGSMITH_PROJECT not set, but tracing is enabled.")

# --- Helper to get LLM settings ---
def get_llm_config() -> dict:
    return {
        'model': OPENAI_MODEL_NAME,
        'base_url': OPENAI_BASE_URL,
        'api_key': OPENAI_API_KEY,
        'temperature': MODEL_TEMPERATURE,
        'streaming': MODEL_STREAMING,
    }

def filter_response(response: str) -> str:
    """Remove thinking tokens and clean up model responses"""
    import re
    
    # Remove <think> blocks
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Remove leading/trailing whitespace
    response = response.strip()
    
    # Remove "Transferring back to supervisor" messages
    response = re.sub(r'Transferring back to supervisor.*?supervisor', '', response, flags=re.DOTALL)
    
    return response