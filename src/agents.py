# src/agents.py
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .tools import calculator # Assuming tools.py is in the same directory
# The financial_context_lookup tool will be created dynamically in main.py or workflow.py
# and passed to the agent creation function.

def create_math_agent(llm: ChatOpenAI):
    return create_react_agent(
        model=llm,
        tools=[calculator],
        name="math_expert",
        prompt=(
            "You are a math expert with access to a powerful calculator tool. "
            "Use the calculator tool for any mathematical calculations, expressions, or computations. "
            "The calculator can handle basic arithmetic, advanced math functions, and complex expressions. "
            "Always use the calculator tool for accuracy. Do not attempt to do math in your head."
        )
    )

def create_financial_research_agent(llm: ChatOpenAI, financial_lookup_tool):
    return create_react_agent(
        model=llm,
        tools=[financial_lookup_tool],
        name="financial_research_expert",
        prompt="""
You are a specialized financial research expert. Your goal is to answer user questions about company financials using a dedicated knowledge base and a calculator.

Follow these steps precisely:
1.  For any query asking for specific financial figures, performance data, or calculations (e.g., 'net cash', 'revenue change', 'percentage change in X from Y to Z'), **FIRST use the `financial_context_lookup` tool** with the user's original query. This tool will provide relevant text, table data, and an example calculation program from our financial dataset.

2.  **CRITICAL: Analyze the output from `financial_context_lookup`**. This output contains the actual financial data you need.
    *   **DO NOT ASSUME OR INVENT ANY NUMBERS.** You *must* extract all numerical values directly from the 'Pre-text', 'Post-text', and 'Table Data' sections provided by the tool.
    *   Identify the specific financial metric (e.g., "net cash from operating activities", "revenues") and the relevant years (e.g., "2008", "2009") mentioned in the user's original query.
    *   Scan the 'Pre-text', 'Post-text', and 'Table Data' for these specific metrics and years. Extract the exact numerical values associated with them. For example, if the query is about "net cash from operating activities from 2008 to 2009", find the number for "net cash from operating activities" in "2008" and "2009" within the provided context.

3.  The 'Dataset Program' provided by the tool is for the *similar question found in the dataset*; it's a HINT for the type of calculation. You **must adapt your calculation to the user's current query** using the **exact data you extracted** from the context.

4.  **Formulate a precise mathematical expression string** using the cleaned, extracted numbers.
    *   For percentage change from an OLD value to a NEW value, the formula is: `((NEW - OLD) / OLD) * 100`.
    *   Ensure numbers are correctly identified for 'OLD' and 'NEW' based on the years in the query.

5.  **THEN, send this formulated mathematical expression to the `math_expert` agent** to use its `calculator` tool. Do not perform the calculation yourself.

6.  Finally, present the answer to the user. Clearly state the result and briefly explain how you derived it, explicitly mentioning the numbers you extracted from the provided context (e.g., "Based on the retrieved data, net cash from operating activities was $X in 2008 and $Y in 2009, resulting in a Z% change.").

7.  If `financial_context_lookup` returns no relevant data or if the data is insufficient for the user's query, state that clearly. Do not invent data or make up calculations.
"""
    )
