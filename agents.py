from crewai import Agent, Task, Crew
from langchain.tools import Tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import load_tools
from langchain.llms import OpenAI
from typing import List

# Tools
search = DuckDuckGoSearchRun()
math_tool = load_tools(["llm-math"], llm=OpenAI(temperature=0))[0]

# AGENT 1: Relevance Checker
relevance_checker = Agent(
    role="Relevance Checker",
    goal="Check if the user's query is relevant",
    backstory="You ensure only relevant questions are passed for further processing. You flag irrelevant ones.",
    allow_delegation=False,
    verbose=True
)

# AGENT 2: RAG Retriever
rag_retriever = Agent(
    role="RAG Retriever",
    goal="Use RAG to gather relevant data for the user's question",
    backstory="You are a data retriever using a knowledge base to provide accurate context to questions.",
    allow_delegation=True,
    verbose=True
)

# AGENT 3: Hallucination Checker
hallucination_checker = Agent(
    role="Hallucination Verifier",
    goal="Verify the factual correctness of the output and provide a hallucination score out of 100%",
    backstory="You verify AI-generated responses and give a factual accuracy score.",
    allow_delegation=False,
    verbose=True
)

# AGENT 4: Math & Web Assistant
math_web_agent = Agent(
    role="Analytical Assistant",
    goal="Perform math operations, plots, and search the web",
    backstory="You are skilled in calculations, plotting, and internet searches.",
    tools=[math_tool, search],
    allow_delegation=True,
    verbose=True
)

# Function to dynamically create tasks
def create_tasks(user_input: str) -> List[Task]:
    return [
        Task(
            description=f"Check if the following question is relevant: '{user_input}'",
            expected_output="Relevant or Irrelevant",
            agent=relevance_checker
        ),
        Task(
            description=f"If the question is relevant, use Retrieval-Augmented Generation to fetch info on: '{user_input}'",
            expected_output="Contextually retrieved data",
            agent=rag_retriever
        ),
        Task(
            description="Evaluate the output and give a hallucination score in percentage.",
            expected_output="Hallucination score (e.g., 92%) and explanation",
            agent=hallucination_checker
        ),
        Task(
            description=f"If the question involves math, plotting, or web search, handle it: '{user_input}'",
            expected_output="Final result",
            agent=math_web_agent
        )
    ]

# Main interactive function
def main():
    print("🤖 Welcome to the CrewAI Assistant!")
    while True:
        user_input = input("\n❓ Ask something (or type 'exit' to quit): ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        # Create tasks based on current input
        tasks = create_tasks(user_input)

        # Create and run crew
        crew = Crew(
            agents=[relevance_checker, rag_retriever, hallucination_checker, math_web_agent],
            tasks=tasks,
            verbose=True
        )

        print("\n⏳ Processing your query...\n")
        result = crew.run()
        print("\n✅ Final Output:\n", result)

if __name__ == "__main__":
    main()
