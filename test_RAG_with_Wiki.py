import argparse
import json

import requests


def main():
    parser = argparse.ArgumentParser(description="Test RAG_with_Wiki or Wiki through /query.")
    parser.add_argument("--task", choices=["RAG_with_Wiki", "Wiki"], default="RAG_with_Wiki")
    parser.add_argument("--url", default="http://127.0.0.1:8082/query")
    parser.add_argument("--model", default="Gemma 4 31B")
    parser.add_argument("--index", default="Conversation1")
    parser.add_argument("--query", default="¿Quién escribió Don Quijote de la Mancha?")
    parser.add_argument("--retriever", default="http://localhost:7997")
    args = parser.parse_args()

    body = {
        "history": [],
        "input_text": args.query,
        "llm_name": args.model,
        "task_config": args.task,
        "docs_k": 5,
        "temp": 0.1,
        "top_p": 0.95,
        "max_tokens": 512,
        "index_name": args.index if args.task == "RAG_with_Wiki" else None,
        "retriever_address": args.retriever,
        "system_prompt": "Responde en español. Si la evidencia proporcionada es insuficiente, indica que no lo sabes.",
    }
    response = requests.post(
        args.url, data={"body": json.dumps(body)}, timeout=(10, 1800)
    )
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result.get("error") or not result.get("text", "").strip():
        raise SystemExit(result.get("error") or "The API returned an empty answer.")


if __name__ == "__main__":
    main()
