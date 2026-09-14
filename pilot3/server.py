from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pipeline import RAGPipeline

app = FastAPI(title="RAG Pipeline API")
pipeline = RAGPipeline(default_llm="salamandra")


class QueryRequest(BaseModel):
    dialog_history: list[str]
    retriever_type: str = "finetuned"
    llm_type: str = "salamandra"
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class SwitchLLMRequest(BaseModel):
    llm_type: str


class QueryResponse(BaseModel):
    query: str
    response: str
    retrieved_documents: dict


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not request.dialog_history:
        raise HTTPException(status_code=400, detail="dialog_history must not be empty")
    result = pipeline.run(
        request.dialog_history,
        retriever_type=request.retriever_type,
        llm_type=request.llm_type,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
    )
    return QueryResponse(
        query=result["query"],
        response=result["response"],
        retrieved_documents=result["retrieved_documents"],
    )


@app.post("/switch_llm")
def switch_llm(request: SwitchLLMRequest):
    pipeline.switch_llm(request.llm_type)
    return {"current_llm": pipeline.current_llm}


@app.get("/health")
def health():
    return {"status": "ready", "current_llm": pipeline.current_llm}
