# Add this at the VERY TOP of agent.py (before any other imports)
import sys
import os
from pathlib import Path

# Get the absolute path to the project root (accountant folder)
project_root = Path(__file__).parent.parent  # Goes from src/ to accountant/
sys.path.insert(0, str(project_root))


# load librarys
from langchain_core.messages import AIMessage, HumanMessage,SystemMessage

from typing_extensions import TypedDict
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
import streamlit as st
from typing import Annotated
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from utils.web_search_tool import search
from utils.rag_web_base_loader_tool import web_loader_tool
from utils.finance_law import finance_law_tool
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from utils.cgnc import cgnc_tool
#from utils import search, web_loader_tool


#load llm and his key
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm_groq=ChatGroq(model="openai/gpt-oss-20b")

## Reducers and creating state
class State(TypedDict):
    messages:Annotated[list,add_messages]


#initiating tools

tools=[cgnc_tool,search,web_loader_tool,finance_law_tool]
llm_with_tools=llm_groq.bind_tools(tools)
#functions for llms
def superbot(state:State):
    return {"messages":[llm_groq.invoke(state['messages'])]}

ACCOUNTING_KEYWORDS = [
    "cgnc", "comptabilité", "accounting", "morocco", "maroc", 
    "financial", "tax", "fiscal", "plan comptable", "audit",
    "bilan", "compte de résultat", "immobilisation", "amortissement",
    "provision", "stock", "créance", "dette", "CGNC"
]

def tool_calling_llm(state:State):
    prompt = """You are a Moroccan accounting expert assistant. 
    
    TOOL USAGE RULES:
    1. For ANY question about Moroccan accounting, CGNC, or finance in Morocco → use 'cgnc_accounting_tool' FIRST
    2. Only use 'google_search' for current events, news, or non-accounting questions
    3. NEVER answer accounting questions without using a tool first
    
    Examples:
    - "What is CGNC?" → use cgnc_accounting_tool
    - "Comment calculer l'amortissement?" → use cgnc_accounting_tool
    - "Latest news in Morocco" → use google_search
    - "Read this webpage" → use rag_web_loader
    
    Current question: {question}
    """
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    # Inject the question into prompt
    formatted_prompt = prompt.replace("{question}", last_message)
    system_message = SystemMessage(content=formatted_prompt)
    all_messages = [system_message] + state["messages"]
    return {"messages":[llm_with_tools.invoke(all_messages)]}


def agent_structring_response(state:State):
    prompt = """You are a response structuring assistant. Follow these rules strictly:
    
    1. If the response came from 'cgnc_accounting_tool', format it professionally as accounting advice
    2. If the response came from 'google_search', cite the source as "Web Search"
    3. If the response is empty, say "I couldn't find specific information on this topic. Try rephrasing your question."
    4. For accounting questions, always mention that the information comes from CGNC standards
    
    Current response to structure: {response}
    """
    
    # Get the last AI message content
    last_response = state["messages"][-1].content if state["messages"] else ""
    formatted_prompt = prompt.replace("{response}", last_response)
    system_message = SystemMessage(content=formatted_prompt)
    
    # Combine system message with existing messages
    all_messages = [system_message] + state["messages"]
    return {"messages":[llm_with_tools.invoke(all_messages)]}

def route_by_keyword(state: State) -> dict:
    """Pre-process to force accounting tool for relevant queries"""
    last_message = state["messages"][-1].content.lower() if state["messages"] else ""
    
    # Check for accounting keywords
    accounting_keywords = ["cgnc", "comptabilité", "accounting", "maroc", "morocco", 
                          "financial", "tax", "fiscal", "amortissement", "bilan"]
    
    if any(keyword in last_message for keyword in accounting_keywords):
        # Force the model to use accounting tool
        forced_prompt = """IMPORTANT: This is an accounting question about Morocco. 
        You MUST use the 'cgnc_accounting_tool' to answer this question.
        Do not use any other tool until you've tried this one.
        
        Question: {question}"""
        formatted_prompt=forced_prompt.replace("{question}", last_message)
        system_message = SystemMessage(content=formatted_prompt)
        return {"messages": [system_message] + state["messages"]}
    
    return state

# initiating state
graph = StateGraph(State)

## Nodes
graph.add_node("tool_calling_llm", tool_calling_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_node("structures", agent_structring_response)
graph.add_node("route_by_keyword", route_by_keyword)

## Edges
graph.add_edge(START, "route_by_keyword")
graph.add_edge("route_by_keyword", "tool_calling_llm")

# First conditional: from tool_calling_llm
graph.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

# After tools, always go to structures
graph.add_edge("tools", "structures")

# Second conditional: from structures
graph.add_conditional_edges(
    "structures",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

# Compile
graph_builder=graph.compile()

#streamlit setup
st.title("Simple LangGraph Test")


test_message = st.chat_input("enter your querry")
# Button to run test
if test_message:
    with st.spinner("Running..."):       
        st.write(test_message)
        result = graph_builder.invoke({'messages': HumanMessage(content=test_message)})
        # chekking tool caling if and else 
        if result["messages"][1].tool_calls:
            for tool_call in result["messages"][1].tool_calls:
                print(f"🔧 Tools were called : {tool_call['name']}")
                st.write(f"the sourse  : {tool_call['name']}")
        
        else:
            st.write("🤖 Source: LLM (no tools used)")
            print(f"🔧 Tools were called : LLM (no tools used)")
        # Display results
        st.markdown(f"📝 Response: {result['messages'][-1].content}")

