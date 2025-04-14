from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI()

# Example data storage (in-memory)
items = []

# Pydantic model for creating items
class Item(BaseModel):
    id: int
    name: str
    description: Optional[str] = None

@app.get("/search", response_model=Optional[Item])
async def search_item(name: str):
    """
    Endpoint to search for an item by name.
    """
    for item in items:
        if item.name == name:
            return item
    return None

@app.post("/create", response_model=Item)
async def create_item(item: Item):
    """
    Endpoint to create a new item.
    """
    if any(existing_item.id == item.id for existing_item in items):
        raise HTTPException(status_code=400, detail="Item with this ID already exists.")
    
    items.append(item)
    return item

# Sample request and response for /search
# Request: GET /search?name=Sample Item
# Response: { "id": 1, "name": "Sample Item", "description": "A sample item" }

# Sample request and response for /create
# Request: POST /create
# Body: { "id": 1, "name": "New Item", "description": "Description of the new item" }
# Response: { "id": 1, "name": "New Item", "description": "Description of the new item" }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
