from typing import Annotated , Sequence ,TypedDict
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage
from langchain_core.messages import ToolMessage
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph,START,END
from langgraph.prebuilt import ToolNode
import google.generativeai as genai

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] =  gemini_api_key


genai.configure(api_key=gemini_api_key)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a:int , b:int):
    """This is an function to add two integers"""
    return a+b

@tool
def subtract(a:int , b:int):
    """This function is used for subtracting two integets"""
    return a-b


@tool
def multiply(a:int, b:int):
    """This function is used for multiplying two integers"""
    return a*b

tools = [add,subtract,multiply]


model = ChatGoogleGenerativeAI(model="gemini-1.5-flash").bind_tools(tools)



def model_call(state:AgentState)->AgentState:
    system_prompt = SystemMessage(
        content="you are an ai assistant ,answer my query using your ability"
    )
    response = model.invoke([system_prompt]+state["messages"])
    return {"messages":response}


def should_continue(state:AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    

graph = StateGraph(AgentState)

graph.add_node("llm_agent",model_call)

tool_node = ToolNode(tools = tools)

graph.add_node("tools",tool_node)

graph.set_entry_point("llm_agent")

graph.add_conditional_edges(
    "llm_agent",
    should_continue,
    {
        "continue":"tools",
        "end":END,
    },
)


graph.add_edge("tools","llm_agent")



app = graph.compile()


def print_stream(stream):
    for s in stream:
        messages = s["messages"][-1]
        if isinstance(messages,tuple):
            print(messages)
        else:
            messages.pretty_print()

inputs = {"messages":[("user","add 10 +20 and then multiply the result by 6 and then tell the future of ai")]}

print_stream(app.stream(inputs,stream_mode="values"))


