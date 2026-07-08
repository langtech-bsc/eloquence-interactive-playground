import pytest
from unittest.mock import MagicMock, patch

from dialogue_manager.dialogue_manager import DialogManager
import dialogue_manager.server as server
import dialogue_manager.client as client


@pytest.fixture
def dm():

    with patch.object(DialogManager, "_load_sessions"), patch.object(DialogManager, "_save_sessions"), patch.object(
        DialogManager, "_init_prompts"
    ), patch.object(DialogManager, "_build_workflow"):

        manager = DialogManager()
        manager.llm = MagicMock()
        return manager


def test_session_creation(dm):

    state = {
        "turn_count": 0,
        "entities": {},
        "history": [],
        "category": None,
        "last_query": None,
        "last_user_utterance": None,
        "terminated": False,
        "ood": False,
    }
    dm.create_session("abc", state)
    assert dm.session_exists("abc")


def test_add_history(dm):

    dm.sessions["1"] = {"history": []}
    dm.add_to_history("1", "CALLER", "hello")
    assert dm.sessions["1"]["history"][0]["text"] == "hello"


def test_reset_session(dm):

    dm.sessions["1"] = {}
    assert dm.reset_session("1") == True
    assert dm.sessions["1"]["turn_count"] == 0


def test_detect_category(dm):

    fake = MagicMock()
    fake.content = "fever"

    dm.llm.invoke.return_value = fake
    dm.category_prompt = MagicMock()
    dm.category_prompt.format.return_value = "prompt"
    result = dm._detect_category("My child has fever")

    assert result == "fever"


def test_detect_unknown_category(dm):

    fake = MagicMock()
    fake.content = "something"

    dm.llm.invoke.return_value = fake
    dm.category_prompt = MagicMock()
    dm.category_prompt.format.return_value = "prompt"
    result = dm._detect_category("hello")

    assert result == "other"


def test_session_exists_false(dm):
    assert dm.session_exists("missing") is False


def test_get_session_state(dm):

    state = {"history": [], "entities": {}}
    dm.sessions["1"] = state

    assert dm.get_session_state("1") == state


def test_get_unknown_session(dm):
    assert dm.get_session_state("unknown") is None


def test_update_last_query(dm):

    dm.sessions["1"] = {"last_query": None}
    dm.update_last_query("1", "Question")

    assert dm.sessions["1"]["last_query"] == "Question"


def test_session_count(dm):
    dm.sessions = {"1": {}, "2": {}}

    assert dm.get_session_count() == 2
