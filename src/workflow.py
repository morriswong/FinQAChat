# src/workflow.py
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class WorkflowState:
    messages: List[Dict[str, str]]
    next_agent: str = "financial_research_expert"

def create_application_workflow(llm: ChatOpenAI, math_agent, financial_agent):
    """Create a custom workflow that works reliably with small models"""
    
    def financial_node(state: WorkflowState) -> WorkflowState:
        """Handle financial research"""
        try:
            config = {"configurable": {"thread_id": "workflow"}}
            result = financial_agent.invoke({"messages": state.messages}, config)
            
            response_content = result['messages'][-1].content if result['messages'] else ""
            
            # Add response to messages
            new_messages = state.messages + [{"role": "assistant", "content": response_content}]
            
            # Check if we need math calculation
            if "NEED_MATH_CALCULATION" in response_content:
                return WorkflowState(messages=new_messages, next_agent="math_expert")
            else:
                return WorkflowState(messages=new_messages, next_agent="END")
                
        except Exception as e:
            error_msg = f"Financial agent error: {str(e)}"
            new_messages = state.messages + [{"role": "assistant", "content": error_msg}]
            return WorkflowState(messages=new_messages, next_agent="END")
    
    def math_node(state: WorkflowState) -> WorkflowState:
        """Handle mathematical calculations"""
        try:
            config = {"configurable": {"thread_id": "workflow"}}
            result = math_agent.invoke({"messages": state.messages}, config)
            
            response_content = result['messages'][-1].content if result['messages'] else ""
            new_messages = state.messages + [{"role": "assistant", "content": response_content}]
            
            return WorkflowState(messages=new_messages, next_agent="END")
            
        except Exception as e:
            error_msg = f"Math agent error: {str(e)}"
            new_messages = state.messages + [{"role": "assistant", "content": error_msg}]
            return WorkflowState(messages=new_messages, next_agent="END")
    
    def route_next(state: WorkflowState) -> str:
        """Simple routing logic"""
        return state.next_agent
    
    # Create the workflow graph
    workflow = StateGraph(WorkflowState)
    workflow.add_node("financial_research_expert", financial_node)
    workflow.add_node("math_expert", math_node)
    
    # Set entry point
    workflow.add_edge(START, "financial_research_expert")
    
    # Add conditional routing
    workflow.add_conditional_edges(
        "financial_research_expert",
        route_next,
        {"math_expert": "math_expert", "END": END}
    )
    
    workflow.add_edge("math_expert", END)
    
    # Compile with memory
    memory = MemorySaver()
    compiled_workflow = workflow.compile(checkpointer=memory)
    
    # Add visualization method
    def visualize_workflow():
        """Generate and display the workflow diagram"""
        try:
            from IPython.display import Image, display
            display(Image(compiled_workflow.get_graph().draw_mermaid_png()))
        except ImportError:
            print("IPython not available. Install with: pip install ipython")
        except Exception as e:
            print(f"Visualization error: {e}")
            print("You can also get the mermaid diagram as text:")
            try:
                print(compiled_workflow.get_graph().draw_mermaid())
            except Exception as e2:
                print(f"Could not generate mermaid diagram: {e2}")
    
    # Attach the visualization method to the compiled workflow
    compiled_workflow.visualize = visualize_workflow
    
    return compiled_workflow