import re
import os
import base64
import json
from typing import Optional, Any, List

from bs4 import BeautifulSoup
from pydub import AudioSegment
from io import BytesIO

from settings import settings
ANONYMOUS_USER = "anonymous"


def detect_audio_format(data: bytes) -> str:
    if data.startswith(b'RIFF') and data[8:12] == b'WAVE':
        return "wav"
    elif data.startswith(b'OggS'):
        return "ogg"
    elif data.startswith(b'fLaC'):
        return "flac"
    elif data.startswith(b'ID3') or data[:2] == b'\xff\xfb':
        return "mp3"
    elif len(data) > 12 and data[4:8] == b'ftyp':
        # ISO Base Media (mp4/m4a); ffmpeg can autodetect with mp4
        return "mp4"
    elif data.startswith(b'\x1A\x45\xDF\xA3'):
        # This could be WebM or Matroska; you'd need deeper parsing to confirm
        return "webm"
    elif data.startswith(b'\x52\x49\x46\x46') and data[8:12] == b'AVI ':
        return "avi"  # Not audio-only but sometimes confused
    else:
        return "unknown"


def bytes_to_wav(audio_bytes, original_format):
    # Let ffmpeg autodetect if format is unknown/None
    if not original_format or original_format == "unknown":
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
    else:
        audio = AudioSegment.from_file(BytesIO(audio_bytes), format=original_format)
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()


def check_llm_interface(llm: str, interface: str, available_llms) -> bool:
    """Checks if the given LLM supports the specified interface."""
    for supported_llm in available_llms.values():
        if supported_llm["display_name"].lower() in llm.lower():
            return supported_llm["interface"] == interface


def encode_audio_stream(audio):
    try:
        audio = bytes(audio)
        audio_format = detect_audio_format(audio)
        if audio_format != "wav":
            audio = bytes_to_wav(audio, audio_format)
        encoded = base64.b64encode(audio).decode("utf-8")
        return encoded
    except Exception as e:
        import sys
        print(e, file=sys.stderr)
        return ""


def replace_doc_links(text):
    def repl(match):
        doc_id = match.group(1)
        url = f"#{doc_id}"
        return f'<a href="{url}" onmouseover="document.getElementById(\'doc_{doc_id}\').style=\'border: 2px solid white;background:#f27618\'; display: block;" onmouseout="document.getElementById(\'doc_{doc_id}\').style=\'border: 1px solid white; background: none; display:none;\'" >[{doc_id}]</a>'
    
    rep = re.sub(r"\[doc ?(\d+)\]", repl, text)
    rep = re.sub(r"\[document ?(\d+)\]", repl, rep)
    rep = re.sub(r"\(doc ?(\d+)\)", repl, rep)
    rep = re.sub(r"\(document ?(\d+)\)", repl, rep)
    rep = re.sub(r"document ?(\d+)", repl, rep)
    rep = re.sub(r"document no. ?(\d+)", repl, rep)
    rep = re.sub(r"Document no. ?(\d+)", repl, rep)

    return rep


def reverse_doc_links(html):
    def repl(match):
        doc_id = match.group(1)
        return f"[doc {doc_id}]"

    # Match <a ...>[n]</a> where n is a number
    return re.sub(r'<a [^>]*href="#(\d+)"[^>]*>\[\d+\]</a>', repl, html)


def _get_user_workspace(user: Optional[str]) -> str:
    """Returns the workspace directory for a given user."""
    return os.path.join(settings.USER_WORKSPACES, user if user else ANONYMOUS_USER)

def _get_user_filepath(user: Optional[str], filename: str) -> str:
    """Constructs the full path to a user-specific file."""
    workspace = _get_user_workspace(user)
    os.makedirs(workspace, exist_ok=True)
    return os.path.join(workspace, filename)

def _load_json(filepath: str, default: Any = []) -> Any:
    """Loads data from a JSON file, returning a default if it doesn't exist."""
    if os.path.exists(filepath):
        with open(filepath, "rt", encoding="utf-8") as f:
            return json.load(f)
    return default

def _save_json(filepath: str, data: Any):
    """Saves data to a JSON file with indentation."""
    with open(filepath, "wt", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def extract_docs_from_rendered_template(rendered_html: str) -> List[str]:
    """Extracts document text from the rendered HTML context."""
    soup = BeautifulSoup(rendered_html, 'html.parser')
    return [div.get_text(strip=True) for div in soup.select('.doc-box')]

def remove_html_tags(text: str) -> str:
    """Removes HTML tags and their content from a string."""
    return re.sub(r'<[^>]*>.*?</[^>]*>', '', text, flags=re.DOTALL)
