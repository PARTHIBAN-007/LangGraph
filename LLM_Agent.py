from typing import TypedDict,List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from dotenv import load_dotenv
import os
load_dotenv()
import google.generativeai as genai

gemini_api_key = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=gemini_api_key)
os.environ["GOOGLE_API_KEY"] = gemini_api_key

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

def process(state:AgentState)->AgentState:
    response = llm.invoke(state["messages"])
    print(f"\n AI :{response.content}")
    return state


graph = StateGraph(AgentState)
graph.add_node("process",process)
graph.add_edge(START,"process")
graph.add_edge("process",END)

agent = graph.compile()


user_input = input("Enter : ")

while user_input!="exit":
    agent.invoke({"messages":[HumanMessage(content=user_input)]})
    user_input = input("Enter : ")
