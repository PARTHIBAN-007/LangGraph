from typing import List,TypedDict 
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


class AgentState(TypedDict):
    messages : List[HumanMessage]

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")


def process(state:AgentState)->AgentState:
    print(state["messages"])
    response = llm.invoke(state["messages"])
    print(f"AI : {response.content}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)


agent = graph.compile()

user_input = input("Enter: ")
while user_input!="exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter : ")
