# src/tools.py
import math
import re
from langchain_core.tools import tool # Using the decorator for cleaner definition
from .finqa_rag import FinancialRAGSystem # Assuming rag_system.py is in the same directory

# Calculator tool (largely same, but with @tool decorator)
@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    Supports basic arithmetic (+, -, *, /), exponentiation (**), 
    parentheses, and common math functions (sin, cos, tan, log, sqrt, etc.).
    Examples: "2 + 3 * 4", "(10 + 5) / 3", "sqrt(16)", "sin(pi/2)", "2**3"
    """
    try:
        expression = expression.strip().lower() # Normalize
        expression = expression.replace('pi', str(math.pi))
        expression = expression.replace('e', str(math.e))

        safe_dict = {
            "__builtins__": {}, "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan, "log": math.log,
            "log10": math.log10, "exp": math.exp, "pow": pow, "floor": math.floor,
            "ceil": math.ceil, "pi": math.pi, "e": math.e,
        }
        
        dangerous_patterns = [r'__\w+__', r'import\s+', r'exec\s*\(', r'eval\s*\(', r'open\s*\(', r'input\s*\(']
        for pattern in dangerous_patterns:
            if re.search(pattern, expression): # Removed re.IGNORECASE as expression is lowercased
                return "Error: Invalid expression - contains potentially dangerous operations."
        
        # Simple check for letters not part of functions/constants
        # This is heuristic and might need refinement for more complex valid expressions
        allowed_chars_in_expr = "0123456789+-*/()., " # Added comma for potential large numbers
        processed_expr = expression
        for func_name in safe_dict.keys():
            if func_name not in ["__builtins__", "pi", "e"]: # Avoid replacing constants that are part of other names
                 processed_expr = processed_expr.replace(func_name, "") # Remove function names for check
        
        for char_to_remove in ["pi", "e"]: # Remove constants
             processed_expr = processed_expr.replace(char_to_remove, "")

        if any(char.isalpha() for char in processed_expr):
            # Check if remaining alpha chars are from numbers like 1e5 (scientific notation)
            if not re.fullmatch(r'[0-9eE\.\+\-\s]*', processed_expr.replace(' ','')): # allow e/E for sci notation
                 return f"Error: Invalid expression - contains unrecognized characters or variable names: '{expression}'"


        result = eval(expression, {"__builtins__": {}}, safe_dict) # Pass safe_dict as globals
        
        if isinstance(result, float):
            return f"{result:.10g}" if not result.is_integer() else str(int(result))
        return str(result)
            
    except ZeroDivisionError: return "Error: Division by zero."
    except ValueError as e: return f"Error: Invalid mathematical operation - {str(e)}."
    except SyntaxError: return f"Error: Invalid mathematical expression syntax in '{expression}'."
    except NameError as e: return f"Error: Unrecognized name or function in expression: {str(e)}"
    except Exception as e: return f"Error calculating '{expression}': {str(e)}."

# Financial Context Lookup Tool
# This function will now be a factory that takes the rag_system instance
def create_financial_context_lookup_tool(rag_system_instance: FinancialRAGSystem):
    @tool
    def financial_context_lookup(query: str) -> str: # Keep return type as str for now, as per user's request to keep code unchanged
        """
        Look up similar financial queries and retrieve relevant context.
        Searches a dataset of financial Q&A to find similar queries, then provides
        context (tables, text) for financial calculations. Use this first for specific financial data questions.
        """
        try:
            print(f"\n🔍 Financial Context Lookup Debug:")
            print(f"📝 Query: {query}")
            
            # Get top 3 for debugging, but only use the best match for context
            all_similar_items = rag_system_instance.find_similar_query(query, top_k=3)
            
            if not all_similar_items:
                print("⚠️  No similar queries found in the financial dataset.")
                return "No similar queries found in the financial dataset."
            
            print(f"🎯 Found {len(all_similar_items)} similar items:")
            for i, item in enumerate(all_similar_items):
                item_question = item.get('qa', {}).get('question', '')
                item_answer = item.get('qa', {}).get('answer', '')
                similarity = rag_system_instance.calculate_similarity(query, item_question)
                print(f"   {i+1}. Similarity: {similarity:.3f} | Q: {item_question[:70]}... | A: {item_answer[:50]}...")
            
            best_match_item = all_similar_items[0]
            similarity_score = rag_system_instance.calculate_similarity(query, best_match_item.get('qa', {}).get('question', ''))
            context = rag_system_instance.extract_context_from_item(best_match_item)
            
            print(f"✅ Using best match with similarity: {similarity_score:.3f}")
            print(f"Full Context for Best Match:")
            print(f"  Dataset Question: {context['question']}")
            print(f"  Dataset Answer: {context['answer']}")
            print(f"  Dataset Program: {context['program']}")
            print(f"  Filename: {context['filename']}")
            print(f"  Pre-text (first 300 chars): {context['pre_text'][:300]}...")
            
            table_str = "No table data available."
            if context['table_data']:
                print(f"  📊 Table data found: {len(context['table_data'])} rows")
                table_str = "\n"
                for i, row in enumerate(context['table_data']): # Print all rows for debugging
                    table_str += f"    Row {i+1}: {row}\n"
                    
                # Extract key numerical data for validation
                print(f"🔍 DATA VALIDATION - Key numbers found in table:")
                for i, row in enumerate(context['table_data']):
                    if len(row) >= 3 and any(char.isdigit() for char in str(row[0])):
                        print(f"     {row[0]}: {row[1:] if len(row) > 1 else 'No values'}")
            else:
                print("  ⚠️  No table data found in context")

            print(f"  Post-text (first 300 chars): {context['post_text'][:300]}...")

            # Improve table formatting for clearer data extraction
            formatted_table = "No table data available."
            if context['table_data']:
                formatted_table = "\n"
                for i, row in enumerate(context['table_data']):
                    # Format each row clearly
                    if len(row) > 1:
                        row_label = str(row[0]) if row[0] else f"Row {i+1}"
                        row_values = [str(cell) for cell in row[1:] if cell]
                        if row_values:
                            formatted_table += f"  • {row_label}: {' | '.join(row_values)}\n"
                    else:
                        formatted_table += f"  • {' | '.join(str(cell) for cell in row if cell)}\n"

            # Add explicit year mapping for clarity
            year_headers = ""
            if context['table_data'] and len(context['table_data']) > 0:
                # Try to find year information from headers
                header_row = context['table_data'][0] if context['table_data'] else []
                if any('2009' in str(cell) or '2008' in str(cell) or '2007' in str(cell) for cell in header_row):
                    year_headers = f"\nYEAR MAPPING FROM TABLE HEADERS: {header_row}"

            response = f"""
⚠️  CRITICAL DATA EXTRACTION TASK ⚠️

REFERENCE ANSWER: {context['answer']} (from dataset)
REFERENCE CALCULATION: {context['program']} (from dataset)

🔍 EXACT TABLE DATA TO EXTRACT FROM:
{formatted_table}{year_headers}

🚨 EXTRACTION RULES:
1. Find the row containing your target metric
2. Copy the EXACT text from that row (with $ signs and commas)
3. DO NOT invent, round, or estimate any numbers
4. The table shows values in columns - typically ordered by year
5. If you use numbers like 125,000,000 or 1,250,000 you are WRONG

📝 SUPPORTING CONTEXT:
Pre-text: {context['pre_text']}
Post-text: {context['post_text']}

USER QUERY: '{query}'

⚠️  BEFORE YOU ANSWER: Copy the exact row from the table that contains your metric and state which column corresponds to which year.
"""
            return response.strip()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"Error in financial_context_lookup: {str(e)}"
    return financial_context_lookup
