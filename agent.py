from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

model = ChatAnthropic(model_namet="claude-3-5-sonnet-20240620") # type: ignore

tools = [TavilySearchResults(max_results=2)]

graph = create_react_agent(model, tools)