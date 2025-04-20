import os
from typing import Dict, List, Optional
import logging
from dotenv import load_dotenv
from openai import OpenAI
from multimodal_retriever import MultimodalRetriever
import json
from sympy import sympify

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Calculator Tool Schema ---
# Define the schema for the calculator tool, similar to the example
calculator_tool_schema = {
    "type": "function",
    "function": {
        "name": "calculate_expression",
        "description": (
            "Evaluates mathematical expressions provided as a string. "
            "Use this for financial calculations like percentage growth ((new-old)/old*100), "
            "currency conversion (USD to INR: amount * 83.5, INR to USD: amount / 83.5), "
            "or other financial ratios based on numbers extracted from the context. "
            "Ensure numbers in the expression do not contain commas."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical formula to evaluate, using standard operators. "
                        "Numbers should not have commas. Use '*' for multiplication, '/' for division. "
                        "Example for growth: (11372538-1024060)/1024060*100. "
                        "Example for USD to INR: 1500*83.5"
                    )
                }
            },
            "required": ["expression"],
            "additionalProperties": False # Prevent unexpected arguments
        }
    }
}

class RAGGenerator:
    """
    Connects the multimodal retrieval system with OpenAI LLM for generating responses,
    including function calling for calculations.
    """
    def __init__(self):
        load_dotenv()
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
        
        # Initialize retriever
        self.retriever = MultimodalRetriever()
        
        # Default model
        self.model = "gpt-4o-mini"
        
        # Initialize conversation history
        self.conversation_history = []
        self.max_history_length = 10  # Store up to 10 exchanges
    
    # --- Calculator Function Implementation ---
    def _calculate_expression(self, expression: str) -> str:
        """
        Safely evaluates mathematical expressions using SymPy.
        Handles basic units like M (million) and K (thousand).
        """
        logger.info(f"Calculating expression: {expression}")
        try:
            # Clean the expression: remove commas, spaces, handle units
            # Be cautious with unit replacement order if complex units are added
            expr_cleaned = (expression
                            .replace(',', '')
                            .replace(' ', '')
                            .replace('$', '') # Remove currency symbols if present
                            .replace('₹', '')
                            .replace('%', '') # Remove percentage signs if user includes them
                            .upper() # Convert to upper for unit matching
                            .replace('M', '*1e6')
                            .replace('K', '*1e3')
                           )

            # Use sympify for safe evaluation
            result_sympy = sympify(expr_cleaned)
            # Evaluate to a float
            result_float = float(result_sympy.evalf())

            # Format the result (e.g., 2 decimal places)
            # Consider adding context awareness later if formatting needs vary (currency vs percentage)
            formatted_result = f"{result_float:.2f}"
            logger.info(f"Calculation result: {formatted_result}")
            return formatted_result
        except Exception as e:
            logger.error(f"Error calculating expression '{expression}': {e}", exc_info=True)
            return f"Calculation Error: {str(e)}"
    
    def _format_context(self, retrieval_results: Dict) -> str:
        """
        Format retrieved context for the LLM prompt
        
        Args:
            retrieval_results: Results from the multimodal retriever
            
        Returns:
            Formatted context as string
        """
        context = "RELEVANT DOCUMENT SECTIONS:\n\n"
        
        # Add text results
        for i, result in enumerate(retrieval_results['text_results'], 1):
            context += f"TEXT SECTION {i} (Page {result['metadata'].get('page_number', 'unknown')}):\n"
            context += f"{result['text'].strip()}\n\n"
        
        # Add image descriptions if available
        if retrieval_results['image_results']:
            context += "RELEVANT IMAGES:\n"
            for i, result in enumerate(retrieval_results['image_results'], 1):
                context += f"IMAGE {i}: From page {result['page_number']}, path: {result['image_path']}\n"
        
        return context
    
    def _build_prompt(self, query: str, context: str) -> tuple:
        """
        Build a prompt for the LLM including query, context, and tool instructions.
        """
        # --- Updated System Prompt ---
        system_prompt = f"""You are an AI assistant specialized in analyzing and answering questions about corporate documents (like 10k reports).
Use ONLY the information provided in the context below to answer the question.
If the context doesn't contain enough information, acknowledge the limitations.
When referencing information from the document, mention the page number.
Reference relevant images if mentioned in the context.

**Calculations:**
- If the query requires calculations (e.g., percentage growth, financial ratios, currency conversion), you MUST use the 'calculate_expression' tool.
- **Do NOT perform calculations yourself.** Extract the necessary numbers from the context first.
- **Currency:** For USD to INR, use the expression `amount * 83.5`. For INR to USD, use `amount / 83.5`.
- **Growth:** Use the formula `((new_value - old_value) / old_value) * 100`.
- **Tool Input:** Provide the calculation as a single string expression to the tool (e.g., `(1500000-1200000)/1200000*100` or `5000*83.5`). Ensure numbers in the expression do NOT contain commas.
- After the tool provides the result, incorporate it naturally into your final answer.

Make your answers concise and directly address the query using the provided context and tool results."""

        user_prompt_content = f"{context}\n\nQUESTION: {query}\n\nANSWER:"
        return system_prompt, user_prompt_content # Return content directly
    
    def _prepare_messages_with_history(self, system_prompt: str, user_prompt: str) -> List[Dict]:
        """
        Prepare messages for the API call, including conversation history
        
        Args:
            system_prompt: System prompt
            user_prompt: Current user prompt
            
        Returns:
            List of message dictionaries for the API call
        """
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history
        for entry in self.conversation_history:
            messages.append({"role": "user", "content": entry["user"]})
            messages.append({"role": "assistant", "content": entry["assistant"]})
        
        # Add current query
        messages.append({"role": "user", "content": user_prompt})
        
        return messages
    
    def generate_answer(self, query: str, n_text_results: int = 5, n_image_results: int = 3, temperature: float = 0.1) -> Dict:
        """
        Generate an answer using RAG and potentially the calculator tool.
        Handles the two-step API call process for function calling.
        """
        final_answer = f"Sorry, an error occurred while generating the answer." # Default error
        sources = [] # Initialize sources

        try:
            # 1. Get retrieval results
            retrieval_results = self.retriever.hybrid_query(
                query=query,
                n_text_results=n_text_results,
                n_image_results=n_image_results
            )
            sources = self._format_sources(retrieval_results) # Format sources early

            # 2. Format context
            context = self._format_context(retrieval_results)

            # 3. Build prompt components
            system_prompt_content, user_prompt_content = self._build_prompt(query, context)

            # 4. Prepare initial messages for the first API call
            messages = self._prepare_messages_with_history(system_prompt_content, user_prompt_content)
            logger.info("Prepared messages for initial API call.")
            # logger.debug(f"Initial messages: {messages}") # Optional: Log messages if needed

            # --- 5. First API Call (Potential Tool Use) ---
            logger.info("Making initial API call with calculator tool enabled.")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=[calculator_tool_schema], # Pass the tool schema
                tool_choice="auto", # Let the model decide whether to use the tool
                temperature=temperature,
            )

            response_message = response.choices[0].message
            # logger.debug(f"Initial API response message: {response_message}") # Optional: Log response

            # --- 6. Handle Tool Calls (if any) ---
            if response_message.tool_calls:
                logger.info("Tool call detected. Executing calculator function.")
                # Append the assistant's message *with* tool_calls to the history for the second call
                messages.append(response_message) # Appends the assistant's request to use the tool

                # Execute the function(s)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    if function_name != "calculate_expression":
                        logger.error(f"Unexpected function called: {function_name}")
                        # Optionally append an error message or just continue
                        continue # Skip this tool call

                    try:
                        # Parse arguments safely
                        function_args = json.loads(tool_call.function.arguments)
                        expression = function_args.get("expression")

                        if expression:
                            # Call the local function
                            function_response = self._calculate_expression(expression=expression)

                            # Append the tool's response to the message history
                            messages.append(
                                {
                                    "tool_call_id": tool_call.id,
                                    "role": "tool",
                                    "name": function_name,
                                    "content": function_response,
                                }
                            )
                            logger.info(f"Appended tool result for call {tool_call.id}")
                        else:
                            logger.error(f"Missing 'expression' argument for tool call {tool_call.id}")
                            # Append an error message for this specific tool call
                            messages.append({
                                "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                                "content": "Error: Missing 'expression' argument."
                            })

                    except json.JSONDecodeError as e:
                         logger.error(f"Failed to parse tool arguments: {tool_call.function.arguments}. Error: {e}")
                         messages.append({
                             "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                             "content": f"Error: Invalid arguments format - {e}"
                         })
                    except Exception as e:
                         logger.error(f"Error executing tool {function_name}: {e}", exc_info=True)
                         messages.append({
                             "tool_call_id": tool_call.id, "role": "tool", "name": function_name,
                             "content": f"Error executing tool: {e}"
                         })


                # --- 7. Second API Call (Get Final Answer) ---
                logger.info("Making second API call with tool results included.")
                # logger.debug(f"Messages for second call: {messages}") # Optional: Log messages
                second_response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages, # Use the history augmented with tool calls and results
                    temperature=temperature, # Can adjust temperature for final response if needed
                )
                final_answer = second_response.choices[0].message.content
                logger.info("Received final answer after tool execution.")

            else:
                # --- 8. No Tool Call ---
                logger.info("No tool call triggered by the model.")
                final_answer = response_message.content # Use the content from the first response

            # --- 9. Update Persistent Conversation History ---
            # Store only the user query and the *final* assistant answer
            if final_answer: # Ensure we have an answer before appending
                 self.conversation_history.append({
                     "user": query, # Store the original user query
                     "assistant": final_answer
                 })

                 # Limit history size
                 if len(self.conversation_history) > self.max_history_length:
                     self.conversation_history.pop(0)
                 logger.info("Conversation history updated.")
            else:
                 logger.warning("Final answer was empty, not updating history.")


            # --- 10. Return Results ---
            return {
                'query': query,
                'answer': final_answer,
                'sources': sources # Return sources formatted earlier
            }

        except Exception as e:
            logger.error(f"Error in generate_answer: {e}", exc_info=True)
            # Ensure sources is a list even in case of error before formatting
            if not isinstance(sources, list):
                sources = []
            return {
                'query': query,
                'answer': f"Sorry, an error occurred while generating the answer: {str(e)}",
                'sources': sources
            }

    # --- Helper to format sources ---
    def _format_sources(self, retrieval_results: Dict) -> List[Dict]:
        """Formats retrieval results into a list of source dictionaries."""
        sources = []
        try:
            for result in retrieval_results.get('text_results', []):
                sources.append({
                    'type': 'text',
                    'page': result.get('metadata', {}).get('page_number'),
                    'content_preview': result.get('text', '')[:100] + "..."
                })

            for result in retrieval_results.get('image_results', []):
                sources.append({
                    'type': 'image',
                    'page': result.get('page_number'),
                    'path': result.get('image_path')
                })
        except Exception as e:
            logger.error(f"Error formatting sources: {e}")
        return sources

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []
        logger.info("Conversation history has been reset")


# Example usage
if __name__ == "__main__":
    generator = RAGGenerator()
    
    # Example conversation
    print("\n--- Starting conversation ---")
    
    # First query
    result1 = generator.generate_answer(
        "What were the financial highlights from the last fiscal year?",
        n_text_results=4,
        n_image_results=2
    )
    print("\nQUERY 1:")
    print(result1['query'])
    print("\nANSWER 1:")
    print(result1['answer'])
    
    # Second query (follow-up)
    result2 = generator.generate_answer(
        "Can you tell me more about their digital transformation initiatives?",
        n_text_results=4,
        n_image_results=2
    )
    print("\nQUERY 2:")
    print(result2['query'])
    print("\nANSWER 2:")
    print(result2['answer'])
    
    # Third query (follow-up)
    result3 = generator.generate_answer(
        "Who are the key executives mentioned in those initiatives?",
        n_text_results=4,
        n_image_results=2
    )
    print("\nQUERY 3:")
    print(result3['query'])
    print("\nANSWER 3:")
    print(result3['answer'])
    
    # Add a test case for calculation
    print("\n--- Testing Calculation Query ---")
    # Example: Assume context provides these numbers (or they are in history)
    # We are simulating the LLM extracting them and asking for calculation
    calc_query = "If revenue was $1,200,000 last year and $1,500,000 this year, what is the percentage growth?"
    # calc_query = "Convert $5000 USD to INR." # Alternative test
    result_calc = generator.generate_answer(calc_query, n_text_results=2, n_image_results=0) # Less context needed maybe
    print("\nQUERY CALC:")
    print(result_calc['query'])
    print("\nANSWER CALC:")
    print(result_calc['answer'])
    print("\nSOURCES CALC:")
    print(result_calc['sources'])

    generator.reset_conversation()
    print("\n--- Conversation reset ---")