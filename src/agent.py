# Add this at the VERY TOP of agent.py (before any other imports)
import sys
import os
from pathlib import Path

# Get the absolute path to the project root (accountant folder)
project_root = Path(__file__).parent.parent  # Goes from src/ to accountant/
sys.path.insert(0, str(project_root))


# load librarys
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
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
#from utils import search, web_loader_tool


#load llm and his key
load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm_groq=ChatGroq(model="openai/gpt-oss-20b")

## Reducers and creating state
class State(TypedDict):
    messages:Annotated[list,add_messages]


#initiating tools

tools=[search,web_loader_tool]
llm_with_tools=llm_groq.bind_tools(tools)
#functions for llms
def superbot(state:State):
    return {"messages":[llm_groq.invoke(state['messages'])]}


def tool_calling_llm(state:State):
    return {"messages":[llm_with_tools.invoke(state["messages"])]}

from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
prompt="your goal is to take a raw text response and u need to structure informations if you recive empty content you need to mention it "

def agent_structring_response(state:State):
    system_message = SystemMessage(content=prompt)
    
    # Combine system message with existing messages
    all_messages = [system_message] + state["messages"]
    return {"messages":[llm_with_tools.invoke(all_messages)]}

# initiating state
graph=StateGraph(State)

## node
#graph.add_node("SuperBot",superbot)
graph.add_node("tool_calling_llm", tool_calling_llm)
graph.add_node("tools", ToolNode(tools))
graph.add_node("structures", agent_structring_response)

## Edges

graph.add_edge(START, "tool_calling_llm")
graph.add_conditional_edges(
    "tool_calling_llm",
    tools_condition,
)
graph.add_edge("tools", "structures")
graph.add_edge("structures", END)

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

