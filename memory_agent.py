import os
from typing import TypedDict,List,Union
from langchain_core.messages import HumanMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import START,END,StateGraph
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict):
    messages : List[Union[HumanMessage,AIMessage]]


llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

def process(state:AgentState)->AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content = response.content))
    print(f"\n AI: {response.content}")
    print(f"CURRENT STATE: {state["messages"]}")
    return state

graph = StateGraph(AgentState)

graph.add_node("process",process)

graph.add_edge(START,"process")
graph.add_edge("process",END)


agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")

while user_input!="exit":
    conversation_history.append(HumanMessage(content=user_input))
    response = agent.invoke({"messages":conversation_history})
    conversation_history = response["messages"]
    user_input = input("Enter: ")

with open("logging.txt","w") as f:
    f.write("your conversation log: ")

    for message in conversation_history:
        if isinstance(message,HumanMessage):
            f.write(f"You: {message.content}")
        elif isinstance(message,AIMessage):
            f.write(f"AI: {message.content}")
    f.write("End of Conversation")
print("Conversation saved to logging.txt")