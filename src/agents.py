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
            "You are a math expert. ALWAYS use the calculator tool for ALL calculations.\n\n"
            "If you see extracted financial data like '2008 = $181,001' and '2009 = $206,588':\n"
            "1. Remove $ and commas: 181001 and 206588\n"
            "2. Use calculator for: ((206588 - 181001) / 181001) * 100\n"
            "3. Report the percentage result\n\n"
            "NEVER calculate in your head. ALWAYS use the calculator tool."
        )
    )

def create_financial_research_agent(llm: ChatOpenAI, financial_lookup_tool):
    return create_react_agent(
        model=llm,
        tools=[financial_lookup_tool],
        name="financial_research_expert",
        prompt="""
You are a financial data extraction expert. Your ONLY job is to extract data and request calculation from the math expert.

**MANDATORY PROCESS:**

STEP 1: Use `financial_context_lookup` with the user's exact query.

STEP 2: **EXTRACT EXACT DATA** from the tool output:
- Find the row in "TABLE DATA" that mentions your metric
- COPY the entire row text exactly as shown
- Example: "Net cash from operating activities: $206,588 | $181,001 | $174,247"

STEP 3: **IDENTIFY VALUES AND YEARS:**
- Look for year indicators like "2009", "2008", "2007" in table headers
- Match them to the values: typically ordered as 2009, 2008, 2007 (newest first)
- State: "I found the exact row: [copied text]"
- State: "Based on the table structure, 2008 = [value], 2009 = [value]"

STEP 4: **STOP AND REQUEST CALCULATION:**
After extracting data, you MUST say exactly:
"Data extracted. NEED_MATH_CALCULATION."

STOP HERE. Do not calculate. Do not answer the question.

**CRITICAL RULES:** 
- Do NOT perform any calculations yourself
- Do NOT use round numbers like 125,000,000
- ALWAYS end with the transfer request using the exact phrase above
- Your job is data extraction ONLY

**VERIFICATION:**
State: "I extracted these exact values: 2008 = [value], 2009 = [value]"
"""
    )
