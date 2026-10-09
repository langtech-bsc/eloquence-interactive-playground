from typing import *
import math

from pydantic import BaseModel, ConfigDict, model_validator


class SparseAttention(BaseModel):
    indices: List[Tuple[int, int]]
    values: List[float]


class AttentionMessageSpan(BaseModel):
    turn_index: int
    side: Literal[0, 1]
    start: int
    end: int


class SummaryAttention(BaseModel):
    """Token indices always refer to the sidecar's prompt + generated sequence."""

    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str
    summary: str
    prompt_token_text: List[str]
    prompt_offsets: List[Tuple[int, int]]
    prompt_length: int
    summary_token_text: List[str]
    summary_offsets: List[Tuple[int, int]]
    summary_length: int
    attention: SparseAttention
    message_spans: List[AttentionMessageSpan] = []

    @model_validator(mode="after")
    def validate_alignment(self):
        for prefix in ("prompt", "summary"):
            text = getattr(self, prefix)
            length = getattr(self, prefix + "_length")
            pieces = getattr(self, prefix + "_token_text")
            offsets = getattr(self, prefix + "_offsets")
            if length < 0 or len(pieces) != length or len(offsets) != length:
                raise ValueError(f"Attention service returned misaligned {prefix} tokens.")
            if any(not 0 <= start <= end <= len(text) for start, end in offsets):
                raise ValueError(f"Attention service returned invalid {prefix} offsets.")
        if not self.summary.strip():
            raise ValueError("Attention service returned an empty summary.")
        total = self.prompt_length + self.summary_length
        if len(self.attention.indices) != len(self.attention.values):
            raise ValueError("Attention indices and values have different lengths.")
        for (row, col), value in zip(self.attention.indices, self.attention.values):
            if not (0 <= row < total and 0 <= col < total):
                raise ValueError("Attention index is outside the token sequence.")
            if not math.isfinite(value) or value < 0:
                raise ValueError("Attention weights must be finite and non-negative.")
        for span in self.message_spans:
            if span.turn_index < 0 or not 0 <= span.start <= span.end <= len(self.prompt):
                raise ValueError("Invalid conversation message span.")
        return self


class RequestQueryLLM(BaseModel):
    operation: Literal["chat", "summarize"] = "chat"
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
    summary_attention: Optional[SummaryAttention] = None
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
