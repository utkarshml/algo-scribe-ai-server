from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from agent import buildGraph 
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
import uuid
app = FastAPI(title="AI Research Agentic System by langgraph")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class query(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"status": "online", "message": "AI Research Agentic System API via Langgraph"}


from typing import Literal

class UserRequest(BaseModel):
    question_name: str | None = "Not provided"
    description: str | None = "also not provided"
    difficulty: Literal["Easy", "Medium", "Hard"] | None = "Easy"
    user_code: str | None = "not provided"
    language: str | None = "not provied"
    message: str
    isChat : bool


@app.post("/solve")
async def ask(item : UserRequest):
    try:
        if(item.isChat == False):
            inputs = {"messages": [("user", f"Generate a Question Solution note for revison and if you have not provided sufficent data then use query related to this question. This is question name  {item.question_name} and question description {item.description} and difficulty {item.difficulty} and user code {item.user_code} and language {item.language}  and message(user needs update something in question or query) :  {item.message}")]}
        else:
            inputs = {"messages": [("user", f"""I have queary related to this question and if you have not provided sufficent data then ask to user about question . The question name is   {item.question_name} and question description {item.description} and difficulty {item.difficulty} and user code {item.user_code} and language {item.language}  and message(user needs update something in question or query) :  {item.message}.
                                    important : until user not say generate note or give question then don't give any answer(ProblemSchema).
                                     """)]}
        response = buildGraph.stream(inputs ,  stream_mode="values")
        response_list = list(response)
        last_response = response_list[-1]["final_response"] if response_list else None
        return {"response": last_response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 