from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

# Example data storage (in-memory)
items = []

class Request(BaseModel):
    query: str


class Response(BaseModel):
    text: str
    documents: list[str]


@app.post("/respond", response_model=Response)
async def respond(request: Request):
    return Response(text=f"This is a Dialogue Manager response to '{request.query}'. This is retrived from document 1 [doc1]",
                    documents=["This is a sample document 1", "This is another supporting doc"])



# Sample request and response for /search
# Request: GET /search?name=Sample Item
# Response: { "id": 1, "name": "Sample Item", "description": "A sample item" }

# Sample request and response for /create
# Request: POST /create
# Body: { "id": 1, "name": "New Item", "description": "Description of the new item" }
# Response: { "id": 1, "name": "New Item", "description": "Description of the new item" }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8088)
