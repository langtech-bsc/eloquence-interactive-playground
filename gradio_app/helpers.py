import re
import base64

from pydub import AudioSegment
from io import BytesIO


from settings import settings, SUPPORTED_LLMS
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
    elif data.startswith(b'\x1A\x45\xDF\xA3'):
        # This could be WebM or Matroska; you'd need deeper parsing to confirm
        return "webm"
    elif data.startswith(b'\x52\x49\x46\x46') and data[8:12] == b'AVI ':
        return "avi"  # Not audio-only but sometimes confused
    else:
        return "unknown"


def bytes_to_wav(audio_bytes, original_format):
    audio = AudioSegment.from_file(BytesIO(audio_bytes), format=original_format)
    wav_io = BytesIO()
    audio.export(wav_io, format="wav")
    return wav_io.getvalue()


def check_llm_interface(llm: str, interface: str) -> bool:
    """Checks if the given LLM supports the specified interface."""
    for supported_llm, supported_interface in SUPPORTED_LLMS.items():
        if supported_llm.lower() in llm.lower():
            return interface in supported_interface


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
    rep = re.sub(r"\[(\d+)\]", repl, rep)
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
