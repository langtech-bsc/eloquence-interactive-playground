from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from dialog_manager import DialogManager, DialogState
import uvicorn
import argparse
from settings import settings

app = FastAPI()

# Initialize dialog manager
dialog_manager = DialogManager()


class Request(BaseModel):
    user_input: str
    dialog_history: Optional[List[Dict[str, str]]] = []
    session_id: str = "default"


class Response(BaseModel):
    response: str
    entities: Dict[str, List[str]]
    category: Optional[str]
    turn_count: int
    terminated: bool
    ood: bool
    session_id: str


class SessionState(BaseModel):
    session_id: str
    state: Dict[str, Any]
    history: List[Dict[str, str]]
    entities: Dict[str, List[str]]
    category: Optional[str]
    turn_count: int
    terminated: bool


class SummaryResponse(BaseModel):
    session_id: str
    summary: str
    entities: Dict[str, List[str]]
    history: List[Dict[str, str]]


@app.post("/process", response_model=Response)
async def process_dialog(request: Request):
    """Process user input and generate system response."""
    try:
        # Initialize or get session state
        if not dialog_manager.session_exists(request.session_id):
            initial_state = DialogState(
                turn_count=0,
                entities={},
                history=[],
                category=None,
                last_query=None,
                last_user_utterance=None,
                terminated=False,
                ood=False,
            )
            dialog_manager.create_session(request.session_id, initial_state)

            # Add welcome message if no history
            if not request.dialog_history:
                welcome = "Hello, this is the pediatric medical call center. How can I help you?"
                dialog_manager.add_to_history(request.session_id, "NURSE", welcome)
                dialog_manager.update_last_query(request.session_id, welcome)

        # Process the user input
        response, state = dialog_manager.process_input(
            session_id=request.session_id, user_input=request.user_input, dialog_history=request.dialog_history
        )

        return Response(
            response=response,
            entities=state.get("entities", {}),
            category=state.get("category"),
            turn_count=state.get("turn_count", 0),
            terminated=state.get("terminated", False),
            ood=state.get("ood", False),
            session_id=request.session_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state/{session_id}", response_model=SessionState)
async def get_state(session_id: str):
    """Get current dialog state for a session."""
    state = dialog_manager.get_session_state(session_id)
    if state:
        return SessionState(
            session_id=session_id,
            state=state,
            history=state.get("history", []),
            entities=state.get("entities", {}),
            category=state.get("category"),
            turn_count=state.get("turn_count", 0),
            terminated=state.get("terminated", False),
        )
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/reset/{session_id}")
async def reset_session(session_id: str):
    """Reset dialog state for a session."""
    success = dialog_manager.reset_session(session_id)
    if success:
        return {"message": f"Session {session_id} reset successfully"}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.post("/end/{session_id}", response_model=SummaryResponse)
async def end_session(session_id: str):
    """End a session and generate summary."""
    summary, state = dialog_manager.end_session(session_id)
    if summary:
        return SummaryResponse(
            session_id=session_id, summary=summary, entities=state.get("entities", {}), history=state.get("history", [])
        )
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "session_count": dialog_manager.get_session_count()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint")
    args = parser.parse_args()
    if args.endpoint is None:
        endpoint = settings.DM_ENDPOINT.replace("http://", "").split(":")
    else:
        endpoint = args.endpoint.replace("http://", "").split(":")
    uvicorn.run(app, host=endpoint[0], port=int(endpoint[1]))
