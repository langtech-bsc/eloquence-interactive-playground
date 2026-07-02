import logging
import time

import openai
import tenacity
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages
from gradio_app.helpers import reverse_doc_links, encode_audio_stream
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = Environment(loader=FileSystemLoader("gradio_app/templates"))
context_template = env.get_template("context_template.j2")


class BSCInteractor:
    def __init__(
        self, api_endpoint, model_name, api_key=None, max_tokens=None, temperature=None, top_p=None, stream=False
    ):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.generate_kwargs = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)
        self.stream = stream
        self.client = openai.OpenAI(base_url=self.api_endpoint, api_key=api_key)
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)

    def __call__(self, documents, history, llm, system_prompt, audio=None):
        messages = self.build_messages(documents, history, llm, system_prompt, audio)
        return self.chat_completion(messages)

    def build_messages(self, documents, history, llm, system_prompt, audio):
        context = ""
        while len(documents) > 0:
            context = context_template.render(documents=documents)
            messages = self._construct_message_list(llm, system_prompt, context, history, audio)
            try:
                num_tokens = apx_num_tokens_from_messages(messages)  # todo for HF, it is approximation
            except:
                num_tokens = len(str(messages).split()) * 2
            if num_tokens + 512 < settings.LLM_CONTEXT_LENGHTS[llm]:
                break
            documents.pop()
        messages = self._construct_message_list(llm, system_prompt, context, history, audio)
        return messages

    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        raise NotImplementedError

    def chat_completion(self, messages):
        logger.info(f"Sending request to {self.model_name} stream={self.stream} ...")
        t1 = time.time()
        try:
            completion = self._request(messages)
        except:
            logger.error("Failed generating response!")
            return ""

        if self.stream:
            return self._generator(completion)

        t2 = time.time()
        usage = completion.usage
        logger.info(
            f"Received response: {usage.prompt_tokens} in + {usage.completion_tokens} out"
            f" = {usage.total_tokens} total tokens. Time: {t2 - t1:3.1f} seconds"
        )
        return completion.choices[0].message.content

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part.choices[0].delta.get("content", "")

    @staticmethod
    def _generator(completion):
        for part in completion:
            yield BSCInteractor.get_stream_text(part)

    def count_tokens(self, messages):
        return apx_num_tokens_from_messages(messages, self.model_name)

    def set_params(self, **params):
        self.generate_kwargs.update(params)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
    def _request(self, messages):
        logger.info(self.api_endpoint + " " + self.model_name)
        logger.info(len(messages))
        logger.info(str([m["role"] for m in messages]))
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=self.stream, **self.generate_kwargs
        )

        return completion


######################################################################################################


import logging
import time
from typing import List, Dict, Any, Optional, Union
import json

import openai
import tenacity
from jinja2 import Environment, FileSystemLoader

from gradio_app.backend.ChatGptInteractor import apx_num_tokens_from_messages
from gradio_app.helpers import reverse_doc_links, encode_audio_stream
from settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
env = Environment(loader=FileSystemLoader("gradio_app/templates"))
context_template = env.get_template("context_template.j2")


class BSCInteractor:
    """
    Base class for LLM chat interactors.
    (Original BSCInteractor code remains unchanged)
    """

    def __init__(
        self, api_endpoint, model_name, api_key=None, max_tokens=None, temperature=None, top_p=None, stream=False
    ):
        self.model_name = model_name
        self.api_endpoint = api_endpoint
        self.generate_kwargs = {"temperature": temperature, "max_tokens": max_tokens, "top_p": top_p}
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)
        self.stream = stream
        self.client = openai.OpenAI(base_url=self.api_endpoint, api_key=api_key)
        logger.info("Creating with endpoint and name:" + api_endpoint + model_name)

    def __call__(self, documents, history, llm, system_prompt, audio=None):
        messages = self.build_messages(documents, history, llm, system_prompt, audio)
        return self.chat_completion(messages)

    def build_messages(self, documents, history, llm, system_prompt, audio):
        context = ""
        while len(documents) > 0:
            context = context_template.render(documents=documents)
            messages = self._construct_message_list(llm, system_prompt, context, history, audio)
            try:
                num_tokens = apx_num_tokens_from_messages(messages)  # todo for HF, it is approximation
            except:
                num_tokens = len(str(messages).split()) * 2
            if num_tokens + 512 < settings.LLM_CONTEXT_LENGHTS[llm]:
                break
            documents.pop()
        messages = self._construct_message_list(llm, system_prompt, context, history, audio)
        return messages

    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        raise NotImplementedError

    def chat_completion(self, messages):
        logger.info(f"Sending request to {self.model_name} stream={self.stream} ...")
        t1 = time.time()
        try:
            completion = self._request(messages)
        except:
            logger.error("Failed generating response!")
            return ""

        if self.stream:
            return self._generator(completion)

        t2 = time.time()
        usage = completion.usage
        logger.info(
            f"Received response: {usage.prompt_tokens} in + {usage.completion_tokens} out"
            f" = {usage.total_tokens} total tokens. Time: {t2 - t1:3.1f} seconds"
        )
        return completion.choices[0].message.content

    @staticmethod
    def get_stream_text(stream_part):
        return stream_part.choices[0].delta.get("content", "")

    @staticmethod
    def _generator(completion):
        for part in completion:
            yield BSCInteractor.get_stream_text(part)

    def count_tokens(self, messages):
        return apx_num_tokens_from_messages(messages, self.model_name)

    def set_params(self, **params):
        self.generate_kwargs.update(params)

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
    def _request(self, messages):
        logger.info(self.api_endpoint + " " + self.model_name)
        logger.info(len(messages))
        logger.info(str([m["role"] for m in messages]))
        completion = self.client.chat.completions.create(
            model=self.model_name, messages=messages, stream=self.stream, **self.generate_kwargs
        )

        return completion


class NERInteractor:
    """
    Named Entity Recognition Interactor using GLiNER model via API endpoint.
    Follows the same pattern as BSCInteractor but adapted for GLiNER NER tasks.

    GLiNER doesn't require system prompts or special formatting - it just needs
    the text input and entity labels. This class handles sending the text to
    a GLiNER API endpoint and formatting the response.

    Args:
        api_endpoint: The API endpoint URL for the GLiNER service
        model_name: Name of the GLiNER model (e.g., "urchade/gliner_multi-v2.1")
        api_key: Optional API key for authentication
        labels: List of entity labels to recognize
        threshold: Confidence threshold for entity predictions (0.0 to 1.0)
        flat_ner: If True, prevents overlapping entities; if False, allows nested entities
    """

    def __init__(
        self,
        api_endpoint: str,
        model_name: str = "urchade/gliner_multi-v2.1",
        api_key: Optional[str] = None,
        labels: Optional[List[str]] = None,
        threshold: float = 0.5,
        flat_ner: bool = True,
    ):
        self.model_name = model_name
        self.api_endpoint = api_endpoint

        # Default entity labels for GLiNER multi-task NER
        self.labels = labels or [
            "person",
            "organization",
            "location",
            "date",
            "event",
            "product",
            "facility",
            "environment",
            "animal",
            "plant",
            "artifact",
            "building",
            "profession",
            "nationality",
            "religion",
            "title",
            "money",
            "law",
        ]

        self.threshold = threshold
        self.flat_ner = flat_ner

        logger.info(f"Creating NERInteractor with endpoint: {api_endpoint}, model: {model_name}")
        logger.info(f"Labels: {self.labels}")
        logger.info(f"Threshold: {threshold}, Flat NER: {flat_ner}")

        # Initialize OpenAI client for API communication (same as BSCInteractor)
        self.client = openai.OpenAI(base_url=self.api_endpoint, api_key=api_key)

    def __call__(
        self,
        query: str,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        flat_ner: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Main interface for performing NER on input text.
        Follows the same pattern as BSCInteractor.__call__().

        Args:
            query: Input text string to analyze (e.g., "Apple hired John Doe in San Francisco.")
            labels: Optional override for entity labels to recognize
            threshold: Optional override for confidence threshold
            flat_ner: Optional override for flat NER setting

        Returns:
            List of entity dictionaries with text, entity label, and confidence score

        Example:
            >>> ner = NERInteractor(api_endpoint="http://localhost:8000/v1", model_name="gliner")
            >>> entities = ner("Apple hired John Doe in San Francisco.")
            >>> print(entities)
            [{'text': 'Apple', 'entity': 'organization', 'score': 0.9995},
             {'text': 'John Doe', 'entity': 'person', 'score': 0.9892},
             {'text': 'San Francisco', 'entity': 'location', 'score': 0.9999}]
        """
        # Build the request with NER parameters (just text and labels, no prompting)
        messages = self.build_ner_messages(query=query, labels=labels, threshold=threshold, flat_ner=flat_ner)

        # Get completion from the API
        response = self.chat_completion(messages)

        # Parse the response into structured entities
        return self.parse_ner_response(response)

    def build_ner_messages(
        self,
        query: str,
        labels: Optional[List[str]] = None,
        threshold: Optional[float] = None,
        flat_ner: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build the request payload for GLiNER NER API.
        GLiNER doesn't need system/user message formatting - just the text and parameters.

        Args:
            query: Input text to analyze
            labels: Entity labels to look for
            threshold: Confidence threshold
            flat_ner: Whether to use flat NER

        Returns:
            List containing a single message with the NER request payload
        """
        # Use provided parameters or fall back to defaults
        use_labels = labels or self.labels
        use_threshold = threshold if threshold is not None else self.threshold
        use_flat_ner = flat_ner if flat_ner is not None else self.flat_ner

        # Construct the NER request payload
        # GLiNER just needs the text and labels - no system prompt needed
        request_payload = {"text": query, "labels": use_labels, "threshold": use_threshold, "flat_ner": use_flat_ner}

        # Wrap in message format that the API expects
        # The message content is the JSON representation of the NER request
        messages = [{"role": "user", "content": json.dumps(request_payload)}]

        logger.info(f"Built NER request for text of length {len(query)}")
        logger.info(f"Labels: {use_labels}, Threshold: {use_threshold}")

        return messages

    def chat_completion(self, messages: List[Dict[str, Any]]) -> str:
        """
        Send messages to the API and get completion.
        Exact same pattern as BSCInteractor.chat_completion().

        Args:
            messages: List of message dictionaries with NER request

        Returns:
            JSON string response from the API
        """
        logger.info(f"Sending NER request to {self.model_name} ...")
        t1 = time.time()

        try:
            completion = self._request(messages)
        except Exception as e:
            logger.error(f"Failed generating NER response: {e}")
            return ""

        t2 = time.time()
        usage = completion.usage
        logger.info(
            f"Received NER response: {usage.prompt_tokens} in + {usage.completion_tokens} out"
            f" = {usage.total_tokens} total tokens. Time: {t2 - t1:3.1f} seconds"
        )

        return completion.choices[0].message.content

    def count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate token count for messages.
        Same pattern as BSCInteractor.count_tokens().

        Args:
            messages: List of message dictionaries

        Returns:
            Estimated token count
        """
        try:
            return apx_num_tokens_from_messages(messages, self.model_name)
        except:
            # Fallback approximation
            return len(str(messages).split()) * 2

    def set_params(self, **params):
        """
        Update NER parameters.
        Same pattern as BSCInteractor.set_params().

        Args:
            **params: Keyword arguments for parameters to update
                - labels: List of entity labels
                - threshold: Default confidence threshold
                - flat_ner: Flat NER setting
        """
        if "labels" in params:
            self.labels = params["labels"]
            logger.info(f"Updated labels to: {self.labels}")

        if "threshold" in params:
            self.threshold = params["threshold"]
            logger.info(f"Updated threshold to: {self.threshold}")

        if "flat_ner" in params:
            self.flat_ner = params["flat_ner"]
            logger.info(f"Updated flat_ner to: {self.flat_ner}")

    @tenacity.retry(wait=tenacity.wait_exponential(multiplier=1, min=4, max=10), stop=tenacity.stop_after_attempt(3))
    def _request(self, messages: List[Dict[str, Any]]):
        """
        Make the actual API request with retry logic.
        Same pattern as BSCInteractor._request().

        Args:
            messages: List of message dictionaries with NER request

        Returns:
            API completion response
        """
        logger.info(f"API Request - Endpoint: {self.api_endpoint}, Model: {self.model_name}")
        logger.info(f"Request payload: {messages[0]['content'][:200]}...")

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=False,  # NER doesn't need streaming
            temperature=0.0,  # No temperature needed for NER
            max_tokens=1024,
        )

        return completion

    def parse_ner_response(self, response: str) -> List[Dict[str, Any]]:
        """
        Parse the API response into structured NER results.

        Converts the GLiNER API response into the desired entity format:
        [{"text":"Apple","entity":"organization","score":0.9995}, ...]

        Args:
            response: Raw JSON response from the API

        Returns:
            List of formatted entity dictionaries
        """
        if not response:
            logger.error("Empty response from API")
            return []

        try:
            # Parse the JSON response
            result = json.loads(response)

            # Handle different response formats
            entities = []

            # If response is already a list of entities
            if isinstance(result, list):
                entities = result
            # If response has an 'entities' key
            elif isinstance(result, dict) and "entities" in result:
                entities = result["entities"]
            # If response is a single entity
            elif isinstance(result, dict):
                entities = [result]

            # Format entities to desired output structure
            formatted_entities = []
            for entity in entities:
                formatted_entity = {
                    "text": entity.get("text", ""),
                    "entity": entity.get("label", entity.get("entity", "unknown")),
                    "score": round(entity.get("score", 0.0), 6),
                }
                formatted_entities.append(formatted_entity)

            # Sort by score in descending order
            formatted_entities.sort(key=lambda x: x["score"], reverse=True)

            logger.info(f"Successfully parsed {len(formatted_entities)} entities")
            return formatted_entities

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from response: {e}")
            logger.debug(f"Raw response: {response[:200]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing NER response: {e}")
            return []


class GLiNERInteractor(NERInteractor):
    """
    Specialized NER Interactor for GLiNER multi-v2.1 API endpoints.
    Pre-configured with comprehensive entity labels for multi-type recognition.

    Uses the standard GLiNER API format without any prompting - just sends
    the text and labels directly to the model endpoint.
    """

    def __init__(self, api_endpoint: str, threshold: float = 0.5, flat_ner: bool = True):
        # Initialize with GLiNER multi-task labels
        super().__init__(
            api_endpoint=api_endpoint,
            model_name="urchade/gliner_multi-v2.1",
            api_key=api_key,
            labels=[
                "person",
                "organization",
                "location",
                "date",
                "event",
                "product",
                "facility",
                "environment",
                "animal",
                "plant",
                "artifact",
                "building",
                "profession",
                "nationality",
                "religion",
                "title",
                "money",
                "law",
            ],
            threshold=threshold,
            flat_ner=flat_ner,
        )
        logger.info("Initialized GLiNERInteractor for comprehensive entity recognition")


# Example usage
if __name__ == "__main__":
    # Initialize the GLiNER NER interactor
    ner = GLiNERInteractor(api_endpoint="http://localhost:8000/v1", api_key="your-api-key-here", threshold=0.5)

    # Example: Basic NER with the desired input/output format
    query = "Apple hired John Doe in San Francisco."

    # Build the NER request (for debugging)
    messages = ner.build_ner_messages(query)
    print(f"NER Request payload:")
    print(json.dumps(messages, indent=2))

    # This would make the actual API call
    # entities = ner(query)
    # print(f"Query: {query}")
    # print(f"Entities: {json.dumps(entities, indent=2)}")

    # Expected output format:
    # [
    #     {"text": "Apple", "entity": "organization", "score": 0.9995285272598267},
    #     {"text": "hired", "entity": "event", "score": 0.8352586627006531},
    #     {"text": "San Francisco", "entity": "environment", "score": 0.9999494552612305}
    # ]

    # Example with custom labels
    custom_messages = ner.build_ner_messages(
        query="Microsoft acquired GitHub for $7.5 billion in 2018.",
        labels=["organization", "money", "date"],
        threshold=0.8,
    )
    print(f"\nCustom NER Request:")
    print(json.dumps(custom_messages, indent=2))


######################################################################################################


class OlmoInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                q = system_prompt + " Context: " + context + " " + q
            if len(a) != 0:
                messages.append(
                    {
                        "role": "user",
                        "content": q,
                    }
                )
            elif len(a) == 0:
                messages.append({"role": "user", "content": [{"type": "text", "text": q}]})
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class SalamandraInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        for q, a in history:
            if len(a) == 0:  # the last message
                messages.append(
                    {
                        "role": "system",
                        "content": context,
                    }
                )
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class EurollmInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                q = system_prompt + " Context: " + context + " " + q
            if len(a) != 0:
                messages.append(
                    {
                        "role": "user",
                        "content": q,
                    }
                )
            elif len(a) == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": q},
                        ],
                    }
                )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class QwenInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = []
        for q, a in history:
            if len(a) == 0:  # the last message
                q = system_prompt + " Context: " + context + " " + q
            if audio is None or len(a) != 0:
                messages.append(
                    {
                        "role": "user",
                        "content": q,
                    }
                )
            elif len(a) == 0:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            #    {
                            #      "type": "text",
                            #       "text": q
                            #   },
                            {
                                "type": "input_audio",
                                "input_audio": {"data": encode_audio_stream(audio), "format": "wav"},
                            }
                        ],
                    }
                )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class SalamandraInteractor(BSCInteractor):
    def _construct_message_list(self, llm, system_prompt, context, history, audio):
        messages = [
            {
                "role": "system",
                "content": system_prompt + " Context: " + context,
            }
        ]
        for q, a in history:
            messages.append(
                {
                    "role": "user",
                    "content": q,
                }
            )
            if len(a) != 0:  # some of the previous LLM answers
                messages.append(
                    {
                        "role": "assistant",
                        "content": reverse_doc_links(a),
                    }
                )
        return messages


class WhisperInteractor(BSCInteractor):
    def __call__(self, documents, history, llm, system_prompt, audio):
        from io import BytesIO

        audio = BytesIO(audio)
        audio.name = "in.wav"
        transcription = self.client.audio.transcriptions.create(
            file=audio, model=self.model_name, language="es", temperature=0.0
        )
        return transcription.text
