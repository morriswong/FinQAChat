#!/usr/bin/env python3
"""
Script to generate Mermaid code for FinQAChat workflow diagram
"""

import sys
import os

# Add parent directory to path to access src modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from src.config import get_llm_config
from src.finqa_rag import FinancialRAGSystem
from src.tools import create_financial_context_lookup_tool
from src.agents import create_math_agent, create_financial_research_agent
from src.workflow import create_application_workflow
from langchain_openai import ChatOpenAI

def main():
    print("🎨 Generating Mermaid code for FinQAChat Workflow...")
    
    # Set up components (minimal config for visualization)
    llm_config = get_llm_config()
    llm = ChatOpenAI(**llm_config)
    
    # Create a minimal RAG system for visualization
    rag_system = FinancialRAGSystem('../data/train.json')
    financial_tool = create_financial_context_lookup_tool(rag_system)
    
    # Create agents
    math_agent = create_math_agent(llm)
    financial_agent = create_financial_research_agent(llm, financial_tool)
    
    # Create workflow
    app = create_application_workflow(llm, math_agent, financial_agent)
    
    print("📊 Workflow created successfully!")
    print("\n📝 Generating Mermaid diagram code...")
    
    try:
        mermaid_text = app.get_graph().draw_mermaid()
        print("\n" + "="*80)
        print("COPY THE CODE BELOW TO https://mermaid.live/")
        print("="*80)
        print(mermaid_text)
        print("="*80)
        print("\n✅ Copy the code above and paste it into https://mermaid.live/ to see your workflow!")
        
    except Exception as e:
        print(f"❌ Could not generate Mermaid code: {e}")

if __name__ == "__main__":
    main()