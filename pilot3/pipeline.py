import threading

from retriever import Retriever
from generator import Generator, AVAILABLE_MODELS


class RAGPipeline:
    def __init__(
        self,
        n_results: int = 20,
        default_llm: str = "salamandra",
        load_in_4bit: bool = False,  # kept for call-site compatibility; unused (LLM is remote)
    ):
        self.retrievers = {
            "finetuned": Retriever(n_results=n_results, use_finetuned=True),
            "baseline": Retriever(n_results=n_results, use_finetuned=False),
        }
        self.current_llm = default_llm
        self.generator = Generator(model_name=AVAILABLE_MODELS[default_llm])
        self._lock = threading.Lock()

    def _switch_llm_unlocked(self, llm_type: str):
        # The LLM runs remotely, so "switching" just changes which model name we request.
        if llm_type == self.current_llm or llm_type not in AVAILABLE_MODELS:
            return
        print(f"\n[PIPELINE] Switching LLM: {self.current_llm} -> {llm_type}")
        self.generator.model_name = AVAILABLE_MODELS[llm_type]
        self.current_llm = llm_type

    def switch_llm(self, llm_type: str):
        with self._lock:
            self._switch_llm_unlocked(llm_type)

    def run(
        self,
        dialog_history: list[str],
        retriever_type: str = "finetuned",
        llm_type: str = "salamandra",
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
    ) -> dict:
        with self._lock:
            self._switch_llm_unlocked(llm_type)
            retriever = self.retrievers.get(retriever_type, self.retrievers["finetuned"])
            user_query = dialog_history[-1]
            retrieved = retriever.retrieve(dialog_history)
            response = self.generator.generate(
                dialog_history,
                retrieved,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_tokens,
            )
            return {
                "query": user_query,
                "retrieved_documents": retrieved,
                "response": response,
            }
