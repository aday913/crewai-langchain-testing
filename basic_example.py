import os

from crewai import Agent, Task, Crew, Process
from crewai_tools import tool
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load the environment variables
load_dotenv()

# Initialize Gemini API
google_api = ChatGoogleGenerativeAI(
    model="gemini-pro",
    verbose=True,
    google_api_key=os.getenv("GEMINI_API_KEY"),
)

# Initialize OpenAI API
openai_api = ChatOpenAI(
    model="gpt-4o",
    verbose=True,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

first_value = ""
second_value = ""

while type(first_value) != int and type(second_value) != int:
    first_value = input("Enter the first number: ")
    second_value = input("Enter the second number: ")

    try:
        first_value = int(first_value)
        second_value = int(second_value)
    except ValueError:
        print("Please enter a valid integer for both numbers")


@tool("Addition tool")
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


gem_agent = Agent(
    role="Simple mathematician",
    goal="Provide the sum of two numbers when asked",
    backstory="You are a simple mathematician who can add two numbers together.",
    tools=[add],
    llm=google_api,
)

openai_agent = Agent(
    role="Simple mathematician",
    goal="Provide the sum of two numbers when asked",
    backstory="You are a simple mathematician who can add two numbers together.",
    tools=[add],
    llm=openai_api,
)

gem_task = Task(
    description=f"Add the following two numbers together: {first_value} and {second_value}",
    agent=gem_agent,
    expected_output="The sum of the two numbers",
)

openai_task = Task(
    description=f"Add the following two numbers together: {first_value} and {second_value}",
    agent=openai_agent,
    expected_output="The sum of the two numbers",
)

gem_crew = Crew(
    agents=[gem_agent], tasks=[gem_task], verbose=2, process=Process.sequential
)

openai_crew = Crew(
    agents=[openai_agent], tasks=[openai_task], verbose=2, process=Process.sequential
)

print("##################################")
print("USING GEMINI AGENT")
gem_result = gem_crew.kickoff()
print(gem_result)

# print("##################################")
# print("USING OPENAI AGENT")
# openai_result = openai_crew.kickoff()
# print(openai_result)
