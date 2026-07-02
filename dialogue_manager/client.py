import requests
from typing import List, Dict, Optional, Any
from settings import settings


class DialogClient:
    def __init__(self, endpoint=None):
        if endpoint is None:
            endpoint = settings.DM_ENDPOINT
        self.endpoint = endpoint

    def process(
        self, user_input: str, dialog_history: Optional[List[Dict[str, str]]] = None, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send user input to dialog manager and get system response.
        Args:
            user_input (str): User's current utterance
            dialog_history (list, optional): Previous dialog turns
            session_id (str, optional): Session identifier

        Returns:
            dict: System response with 'response', 'entities', 'category',
                  'terminated', 'turn_count' fields
        """
        payload = {
            "user_input": user_input,
            "dialog_history": dialog_history or [],
            "session_id": session_id or "default",
        }

        response = requests.post(f"{self.endpoint}/process", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return {"response": "Error processing request", "error": response.text, "terminated": False}

    def get_state(self, session_id: str) -> Dict[str, Any]:
        """
        Get current dialog state for a session.
        Args:
            session_id (str): Session identifier
        Returns:
            dict: Current dialog state including entities, category, history, turn_count
        """
        response = requests.get(f"{self.endpoint}/state/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}

    def reset(self, session_id: str) -> bool:
        """
        Reset dialog state for a session.
        Args:
            session_id (str): Session identifier
        Returns:
            bool: Success status
        """
        response = requests.post(f"{self.endpoint}/reset/{session_id}")
        return response.status_code == 200

    def end_session(self, session_id: str) -> Dict[str, Any]:
        """
        End the dialog session and get summary of the dialog.
        Args:
            session_id (str): Session identifier
        Returns:
            dict: Summary and final state
        """
        response = requests.post(f"{self.endpoint}/end/{session_id}")
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
