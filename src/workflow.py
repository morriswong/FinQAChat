# src/workflow.py
from langchain_openai import ChatOpenAI
from langgraph_supervisor import create_supervisor
from langgraph.checkpoint.memory import MemorySaver

def create_application_workflow(llm: ChatOpenAI, math_agent, financial_agent): # Pass created agents
    workflow = create_supervisor(
        [financial_agent, math_agent], # Order can influence supervisor choice
        model=llm, # Supervisor can use the same or a different model
        prompt=(
            "You are a team supervisor. Based on the user's query, route it to the appropriate expert:\n"
            "- For specific financial questions requiring data lookup from company reports (e.g., 'net cash', 'revenue change', 'percentage change in X from Y to Z'), use 'financial_research_expert'. This expert will first look up data and then perform calculations if needed.\n"
            "- For direct mathematical calculations or general math problems where numbers are provided or the problem is abstract (e.g., 'what is 5 * 7?', 'solve x + 5 = 10'), use 'math_expert'.\n"
            "If unsure, lean towards 'financial_research_expert' for anything that sounds like it might need specific financial data."
        )
    )
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)