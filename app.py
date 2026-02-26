import os
import streamlit as st
from dotenv import load_dotenv

from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load env
load_dotenv()

# Initialize LLM
llm = ChatOllama(model="llama3", temperature=0.3)

# Initialize Search
search_tool = TavilySearch(max_results=5)

# ----------------------------
# 1️⃣ Define Graph State
# ----------------------------

class GraphState(TypedDict):
    messages: List
    use_search: bool


# ----------------------------
# 2️⃣ Router Node
# ----------------------------

def router(state: GraphState):
    last_message = state["messages"][-1].content
    keywords = ["latest", "news", "recent", "2025", "2026", "current"]

    use_search = any(word in last_message.lower() for word in keywords)

    return {"use_search": use_search}


# ----------------------------
# 3️⃣ Search Node
# ----------------------------

def search_node(state: GraphState):
    query = state["messages"][-1].content
    results = search_tool.invoke(query)

    enriched_prompt = f"""
    Use the following web results to answer:

    {results}

    Question: {query}
    """

    state["messages"].append(HumanMessage(content=enriched_prompt))
    return state


# ----------------------------
# 4️⃣ LLM Node
# ----------------------------

def llm_node(state: GraphState):
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state


# ----------------------------
# 5️⃣ Build Graph
# ----------------------------

graph = StateGraph(GraphState)

graph.add_node("router", router)
graph.add_node("search", search_node)
graph.add_node("llm", llm_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: "search" if state["use_search"] else "llm",
)

graph.add_edge("search", "llm")
graph.add_edge("llm", END)

app_graph = graph.compile()


# ----------------------------
# 6️⃣ Streamlit UI
# ----------------------------

st.set_page_config(page_title="LangGraph Research Assistant")
st.title("🤖 LangGraph Research Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are a helpful research assistant.")
    ]

# Display history
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# Chat input
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.messages.append(HumanMessage(content=prompt))

    result = app_graph.invoke({
        "messages": st.session_state.messages,
        "use_search": False
    })

    st.session_state.messages = result["messages"]

    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[-1].content)
