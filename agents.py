from crewai import Agent, Task, Crew
from langchain_openai import OpenAI
from langchain.tools import tool
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
import numexpr

# Set up LLM
llm = OpenAI(temperature=0)

# Custom Calculator Tool using the numexpr library
class CalculatorTools:
    @tool("Make a calculation")
    def calculate(operation: str) -> str:
        """Useful to perform any mathematical calculations,
        like sum, minus, multiplication, division, etc.
        The input to this tool should be a mathematical
        expression, e.g., '2007' or '5000/210'
        """
        try:
            result = numexpr.evaluate(operation)
            return str(result)
        except Exception as e:
            return f"Error: {e}"

# Create instances of the tools
calculator_tool = CalculatorTools().calculate  # This gets the @tool-decorated method
search_tool = DuckDuckGoSearchResults()  # Use the existing tool directly

# AGENTS
relevance_checker = Agent(
    role="Relevance Checker",
    goal="Check if the user's query is relevant",
    backstory="You ensure only relevant questions are passed for further processing. You flag irrelevant ones.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

rag_retriever = Agent(
    role="RAG Retriever",
    goal="Use RAG to gather relevant data for the user's question",
    backstory="You are a data retriever using a knowledge base to provide accurate context to questions.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

hallucination_checker = Agent(
    role="Hallucination Verifier",
    goal="Verify the factual correctness of the output and provide a hallucination score out of 100%",
    backstory="You verify AI-generated responses and give a factual accuracy score.",
    allow_delegation=True,
    verbose=True,
    llm=llm
)

math_web_agent = Agent(
    role="Analytical Assistant",
    goal="Perform math operations, plots, and search the web",
    backstory="You are skilled in calculations, plotting, and internet searches.",
    tools=[calculator_tool, search_tool],
    allow_delegation=True,
    verbose=True,
    llm=llm
)

# TASKS (rest of your code remains the same)
task1 = Task(
    description="Check if the user question is relevant to the system's domain. If not, mark as irrelevant and halt further processing.",
    expected_output="Relevant or Irrelevant",
    agent=relevance_checker
)

task2 = Task(
    description="If the question is relevant, use Retrieval-Augmented Generation to fetch related information from the knowledge base.",
    expected_output="Contextually retrieved data",
    agent=rag_retriever
)

task3 = Task(
    description="Check the output response and evaluate how factually accurate it is. Give a hallucination score in percentage.",
    expected_output="Hallucination score (e.g., 92%) and explanation",
    agent=hallucination_checker
)

task4 = Task(
    description="If the question involves math, plots, or needs web search, handle it here.",
    expected_output="Final answer with computations or search results",
    agent=math_web_agent
)

# CREW
crew = Crew(
    agents=[relevance_checker, rag_retriever, hallucination_checker, math_web_agent],
    tasks=[task1, task2, task3, task4],
    verbose=True
)

# RUN
result = crew.run(input="What is 2007 + 5000?")
print(result)