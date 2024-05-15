import os

from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
from langchain.agents import Tool
from langchain_community.tools.google_scholar.tool import GoogleScholarQueryRun
from langchain_community.utilities.google_scholar import GoogleScholarAPIWrapper
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


# Initialize Google Scholar API tool
# search = GoogleScholarQueryRun(
#     api_wrapper=GoogleScholarAPIWrapper(serp_api_key=os.getenv("SERPAPI_KEY"))
# )
#
# scholar_tool = Tool(
#     name="Google Scholar Search Tool",
#     description="Search Google Scholar for academic research papers.",
#     func=search.run,
# )

scholar_tool = SerperDevTool(search_url="https://google.serper.dev/scholar")

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
    goal="Find relevant papers from google scholar based on a given input",
    backstory="You are a scholar who can provide information from Google Scholar.",
    tools=[scholar_tool],
    llm=google_api,
)

scholar_task = Task(
    description=f"""
    Find relevant academic research papers from google scholar based on the following input:
    {get_input()}
    """,
    agent=scholar,
    expected_output="A bulleted list of at least 7 relevant papers from Google Scholar",
)

scholar_crew = Crew(
    agents=[scholar], tasks=[scholar_task], verbose=2, process=Process.sequential
)
scholar_result = scholar_crew.kickoff()
print(scholar_result)
