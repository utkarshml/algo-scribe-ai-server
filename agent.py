from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Annotated, Sequence, TypedDict, List
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import Annotated
from langchain_core.messages import SystemMessage , HumanMessage
from langchain_core.tools import tool
from pydantic import BaseModel , Field
from langgraph.graph import StateGraph, END , MessagesState
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langgraph.prebuilt import ToolNode
import requests
from dotenv import load_dotenv
import os
load_dotenv()

class AgentInput(MessagesState):
    pass


llm = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash-preview-05-20",
    temperature=0, # Keep temperature low for structured outputs
    max_tokens=None,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    timeout=None,
    max_retries=2,

    # other params...
)
search = TavilySearchResults(max_results=2)

class ChatOutput(BaseModel):
    """ This is Response for User query """ 
    output: str = Field(..., alias="return", description="The expected return user query answers and doubts.")
    isChat : bool = True

class OutputFormat(BaseModel):
    """ Description of return type for the problem """ 
    return_: str = Field(..., alias="return", description="The expected return type and description of the function output.")

class ProblemSchema(BaseModel):
    """ Schema for a coding problem """
    question_name: str = Field(..., description="Name/title of the coding problem.")
    description: str = Field(..., description="Detailed description of the problem statement.")
    output_format: OutputFormat = Field(..., description="Description of output format of the solution.")
    userCode: str = Field(..., description="C++ function signature for the user to implement.")
    topic: List[str] = Field(..., description="List of DSA topics related to this problem.")
    difficulty: str = Field(..., description="Difficulty level of the problem (e.g., Easy, Medium, Hard).")
    solution_code: str = Field(..., description="Complete C++ solution code for the problem.")
    note: str = Field(..., description="Detailed good markdown note explaining and focusing on the Design Pattern of this problem  and another approaches. what is most rememberable about similar question and should be concise and simple english")
    interview_tips: List[str] = Field(..., description="List of interview tips related to this problem.")
    isChat : bool = False
class AgentState(MessagesState):
    final_response : ProblemSchema | ChatOutput



class AgentOutput(TypedDict):
    messages : ProblemSchema | ChatOutput

tools = [ search , ProblemSchema , ChatOutput]
model = llm.bind_tools(tools, tool_choice="any")


template = """
You are a helpful, knowledgeable, and professional coding teacher.
You are patient, clear, and structured in your teaching.
You specialize in Data Structures, Algorithms, and Problem Solving for coding interviews.
You have access to advanced tools like Travily for internet search to ensure your answers are accurate and up-to-date.
You focus on clarity, precision, and practical guidance for learners preparing for technical interviews.
Your teaching style is methodical, using clean code, proper explanations, and structured revision notes.
You prefer C++ as the default language, but respect user preference if they specify another.
You are concise, professional, and never waste words with greetings or unnecessary flourishes.
"""

config={"configurable":{"thread_id":1}}

def model_call(state:AgentState) -> AgentState:
    response = model.invoke([SystemMessage(content=template)] +state["messages"],config=config)
    return {"messages": [response]}



def should_continue(state:AgentState) -> str:
    """
    Determine whether the agent should continue or end the conversation.
    This function looks at the last message in the state and checks if
    it contains tool calls. If it does, the agent continues, otherwise
    it ends the conversation.
    """
    messages = state["messages"]
    last_message = messages[-1]
    if (
        len(last_message.tool_calls) == 1
        and last_message.tool_calls[0]["name"] == "ProblemSchema"
    ): 
        return "response1"
    elif (
        len(last_message.tool_calls) == 1
        and last_message.tool_calls[0]["name"] == "ChatOutput"
    ):
        return "response2"
    else:
        return "continue"


def response1(state:AgentState) -> AgentState:
    response = ProblemSchema(**state["messages"][-1].tool_calls[0]["args"])
    return {"final_response": response}
def response2(state:AgentState) -> AgentState:
    response = ChatOutput(**state["messages"][-1].tool_calls[0]["args"])
    return {"final_response": response}
    

graph = StateGraph(AgentState,input=AgentInput,output=AgentOutput)
graph.add_node("agent", model_call)
graph.add_node("response1", response1)
graph.add_node("response2", response2)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "response1": "response1",
        "response2": "response2"
    },
)
graph.add_edge("tools", "agent")
graph.add_edge("response2", END)
graph.add_edge("response1", END)
buildGraph = graph.compile()






# def print_stream(stream):
#     for s in stream:
#         message = s["messages"][-1]
#         if isinstance(message, tuple):
#             print(message)
#         else:
#             message.pretty_print()

# inputs = {"messages": [("user", "Generate a travel itinerary for Sydney from 2025-06-01 to 2025-06-10 for 2 travelers with a Luxury style and special interests in Hiking and fight start from Delhi ")]}
# print_stream(app.stream(inputs, stream_mode="values"))
