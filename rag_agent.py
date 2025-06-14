import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph,START,END
from typing import TypedDict,List,Union,Annotated,Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage,BaseMessage,SystemMessage,AIMessage,ToolMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool


load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model = "gemini-1.5-flash")


embeddings = GoogleGenerativeAIEmbeddings(
   model="models/embedding-001"
)

pdf_path = "Stock_Market_Performance_2024.pdf"


if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file is not found : {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)


try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error Loading PDF : {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 10000,
    chunk_overlap = 200
)


pages_split = text_splitter.split_documents(pages)


try:
    vectorstore = FAISS.from_documents(
        documents=pages_split,
        embedding=embeddings,
        collection_name = "stockMarket"

    )
    print(f"Created FAISS DB")
except Exception as e:
    print(f"Error setting up FAISS")
    raise

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k":5,
    }
)

@tool
def retriever_tool(query:str)->str:
    """This tool searches and returns the information from the stock market performance 2024 document"""
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the stock market performance 2024 Document"
    
    results = []
    for i,doc in enumerate(docs):
        results.append(f"Document {i+1}: \n {doc.page_content}")
    return "\n\n".join(results)

tool = [retriever_tool]

llm = llm.bind_tools(tool)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]


def should_continue(state:AgentState):
    """Check if the last message contains calls"""
    result = state["messages"][-1]
    return hasattr(result,'tool_calls') and len(result.tool_calls)>0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name:our_tool for our_tool in tool}

def call_llm(state:AgentState)->AgentState:
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    message = llm.invoke(messages)
    return {"messages":[message]}

def take_action(state:AgentState)->AgentState:
    """Execute tool calls from the LLM response"""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(f"Calling Tool: {t["name"]} with query : {t['args'].get('query','No Query Provided')}")


        if not t['name'] in tools_dict:
            print(f"\n Tool : {t['name']} does not exist")
            result = "Incorrect TOol Name,please retry and select tool from list of available tools"
        else:
            result = tools_dict[t['name'].invoke(t['args'].get('query',''))]
            print(f"Result Length: {len(str(result))}")

        results.append(ToolMessage(tool_call_id = t[id],name = t['name'],content = str(result)))

    print("Tools execution completed.Back to the model")
    return {"messages":results}


graph = StateGraph(AgentState)
graph.add_node("llm",call_llm)
graph.add_node("retriever_agent",take_action)


graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_tool",
     False:END}
)

graph.add_edge("retriever_agent","llm")

graph.set_entry_point("llm")


rag_agent = graph.compile()


def running_agent():
    print("\n =======RAG Agent ============")

    while True:
        user_input = input("\n What is your question: ")
        if user_input.lower() in ["exit","quit"]:
            break

        messages = [HumanMessage(content=user_input)]

        result = rag_agent.invoke({"messages":messages})
        print("\n===ANSWER===")
        print(result['messages'][-1].content)


running_agent()
