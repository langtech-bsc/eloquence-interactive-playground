from typing import *
from pydantic import BaseModel


class RequestQueryLLM(BaseModel):
    history: List[List[str]]
    input_text: str
    llm_name: str
    task_config: Optional[str] = "Basic"
    docs_k: Optional[int]
    temp: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    index_name: Optional[str]
    retriever_address: Optional[str] = "public"
    system_prompt: Optional[str] = None
    language: Optional[str] = None
    render_doc_links: Optional[bool] = True
    chunk_start_seconds: Optional[float] = 0.0


class RequestBatchQuery(BaseModel):
    llm_name: str
    task_config: Optional[str] = "Basic"
    docs_k: Optional[int]
    temp: Optional[float]
    top_p: Optional[float]
    max_tokens: Optional[int]
    index_name: Optional[str]
    retriever_address: Optional[str] = "public"
    system_prompt: Optional[str] = None
    render_doc_links: Optional[bool] = True


class RequestIngest(BaseModel):
    index_name: str
    embed_name: str
    chunk_size: Optional[int] = 500
    percentile: Optional[float] = 0.9
    splitting_strategy: Optional[str] =  "recursive"
    retriever_address: Optional[str] = "public"
    append: Optional[bool] = False
    snippet_metadata: Optional[Dict[str, Any]] = None
    snippet_turns: Optional[List[Dict[str, Any]]] = None


class ResponseQueryLLM(BaseModel):
    text: str
    documents: List[str]
    documents_metadata: Optional[List[Any]] = None
    transcription_metadata: Optional[Dict[str, Any]] = None
    transcription_turns: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class ResponseBatchQuery(BaseModel):
    processed: Dict
    error: Optional[str] = None


class ResponseIngest(BaseModel):
    status: str
    msg: str


class ResponseList(BaseModel):
    available: List[str]


class ResponseFeedback(BaseModel):
    filter: str
    feedback: List[Dict]
