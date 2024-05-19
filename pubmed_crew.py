import os

from crewai import Agent, Task, Crew, Process
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# Load the environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")

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

tool = PubmedQueryRun()


def get_input() -> str:
    print("Insert your text. Enter 'q' or press Ctrl-D (or Ctrl-Z on Windows) to end.")
    contents = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line == "q":
            break
        contents.append(line)
    return "\n".join(contents)


scholar = Agent(
    role="Paper Finder",
    goal="Find relevant papers from google scholar based on a given input and provide a summary.",
    backstory="You are a scholar who can provide information from PubMed.",
    tools=[tool],
    llm=google_api,
)

ambassador = Agent(
    role="Ambassador",
    goal="""Given information from a research paper, provide a summary of the paper that 
    is easy to understand as a grad student""",
    backstory="""
    You are an ambassador who can provide information on the latest research papers to a more general audience.
    """,
    llm=openai_api,
)

scholar_task = Task(
    description=f"""
    Find relevant academic research papers and summarize them from pubmed based on the following input:
    {get_input()}
    """,
    agent=scholar,
    expected_output="A summary of the relevant papers.",
)

ambassador_task = Task(
    description=f"""
    Given information from a research paper, provide a summary of the paper that is easy to understand as a grad student.
    """,
    agent=ambassador,
    expected_output="A summary of the research paper that is easy to understand as a grad student.",
)

scholar_crew = Crew(
    agents=[scholar, ambassador],
    tasks=[scholar_task, ambassador_task],
    verbose=2,
    process=Process.sequential,
)
scholar_result = scholar_crew.kickoff()
print(scholar_result)
