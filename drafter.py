from langchain_core.messages import BaseMessage,SystemMessage,HumanMessage,AIMessage,ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
import os
from dotenv import load_dotenv
from typing import TypedDict , List,Union,Annotated,Sequence
load_dotenv()


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")


document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]



@tool
def update(content:str)->str:
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n {document_content}"


@tool
def save(filename:str)->str:
    """save the current document to a text file and finish the process
    Args:
        filename : Name for the text file
    """

    global document_content

    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"

    try:
        with open(filename,"w") as f:
            f.write(document_content)
        print(f"\n Document has been saved to {filename}")
        return f"Document has been saved successfully to {filename}"
    except Exception as e:
        return f"Error saving docuemnts : {e}"
    
tools = [update,save]

model = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")

def our_agent(state:AgentState)->AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)
    
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)

    print(f"\n AI : {response.content}")
    if hasattr(response,"tool_calls") and response.tool_calls:
        print(f"Using tools : {[tc["name"] for tc in response.tool_calls]}")

    return { "messages": list(state["messages"])+ [user_message,response]}

def should_continue(state:AgentState)->str:
    """Determine if we should continue or end the conversation"""

    messages = state['messages']
    if not messages:
        return "continue"

    for message in reversed(messages):
        if (isinstance(messages,ToolMessage) and "saved" in messages.content.lower() and "document" in message.content.lower()):
            return "end"
    return "continue"

def print_messages(messages):
    """Function i made to print the message in a more readable format"""

    if not messages:
        return 
    
    for message in messages[-3:]:
        if isinstance(message,ToolMessage):
            print(f"\n Tool Result: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent",our_agent)
graph.add_node("tools",ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent","tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue":"agent",
        "end":END,
    }
)

app = graph.compile()


def run_document_again():
    print("\n DRAFT SAVER")

    state  = {"messages": []}

    for step in app.stream(state,stream_mode = "values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n ================DRAFT SAVED =========================")


if __name__ == "__main__":
    run_document_again()

