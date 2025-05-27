# src/main.py
import os # Already imported in config, but good for explicitness

from langchain_openai import ChatOpenAI
from .config import get_llm_config, DATASET_PATH # Use your config module
from .finqa_rag import FinancialRAGSystem
from .tools import create_financial_context_lookup_tool, calculator # calculator is directly usable
from .agents import create_math_agent, create_financial_research_agent
from .workflow import create_application_workflow

def run_chat_interface():
    print("Initializing Financial Chat System...")

    # 1. Initialize LLM
    llm_configs = get_llm_config()
    llm = ChatOpenAI(**llm_configs)

    # 2. Initialize RAG System
    rag_system_instance = FinancialRAGSystem(dataset_path=DATASET_PATH)

    # 3. Create Tools
    # Calculator tool is already defined and decorated with @tool
    # Create the financial lookup tool by passing the RAG instance
    financial_lookup_tool = create_financial_context_lookup_tool(rag_system_instance)

    # 4. Create Agents
    math_agent_instance = create_math_agent(llm)
    financial_agent_instance = create_financial_research_agent(llm, financial_lookup_tool)

    # 5. Create Workflow
    app = create_application_workflow(llm, math_agent_instance, financial_agent_instance)

    print("Welcome to the LangGraph Supervisor Chat!")
    print("Ask financial questions or math problems.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    # Simple unique thread_id per session for memory
    import uuid
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    print(f"Session ID: {session_id}")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if not user_input.strip():
            continue

        try:

            print("Assistant: ", end="", flush=True)  
            for msg, metadata in app.stream({  
                "messages": [{"role": "user", "content": user_input}]  
            }, config, stream_mode="messages"):  
                if hasattr(msg, 'content') and msg.content:  
                    print(msg.content, end="", flush=True)
            print("\n")  # New line after streaming completes  

        except Exception as e:
            print(f"\nError during stream: {e}")
            import traceback
            traceback.print_exc()
            print()

if __name__ == "__main__":
    # This allows running main.py directly from the src directory or project root
    # For running from project root: python -m src.main
    run_chat_interface()