import streamlit as st
import datetime
import wikipedia
import operator
from typing import TypedDict, Annotated, Sequence

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from langchain.tools import tool
from langchain.tools.python.tool import PythonREPLTool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


st.set_page_config(page_title="LangGraph + Ollama (5 Tools)")
st.title("🤖 LangGraph Agent with 5 Tools (Ollama Local)")


llm = ChatOllama(
    model="llama3",
    temperature=0
)



def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


def wiki_search(query: str) -> str:
    """Search Wikipedia for a topic."""
    try:
        return wikipedia.summary(query, sentences=3)
    except:
        return "No results found."


def current_datetime(_: str) -> str:
    """Get current date and time."""
    return str(datetime.datetime.now())


def read_text_file(file_path: str) -> str:
    """Read content from a text file."""
    try:
        with open(file_path, "r") as f:
            return f.read()
    except:
        return "Could not read file."

python_tool = PythonREPLTool()

tools = [
    calculator,
    wiki_search,
    current_datetime,
    read_text_file,
    python_tool
]

tool_node = ToolNode(tools)



class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]



def call_model(state):
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}



def should_continue(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END

workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
)

workflow.add_edge("tools", "agent")

graph = workflow.compile()


if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask something...")

if user_input:
    st.session_state.messages.append(HumanMessage(content=user_input))

    result = graph.invoke({
        "messages": st.session_state.messages
    })

    st.session_state.messages = result["messages"]

# Display Chat
for msg in st.session_state.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    else:
        st.chat_message("assistant").write(msg.content)
