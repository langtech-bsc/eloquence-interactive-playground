import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError


SPEAKER_MATCH_CONFIDENCE = 0.80
SPEAKER_NAME_CONFIDENCE = 0.90
SEMANTIC_SPLIT_CONFIDENCE = 0.90
MAX_RECENT_EXAMPLES = 3
MAX_RECENT_TURNS = 6

logger = logging.getLogger(__name__)


class SpeakerMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    local_speaker: str
    canonical_speaker: Optional[str]
    confidence: float = Field(ge=0.0, le=1.0)
    display_name: Optional[str] = None
    name_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SemanticSplit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_index: int = Field(ge=0)
    after_segment_index: int = Field(ge=0)
    next_speaker: str
    confidence: float = Field(ge=0.0, le=1.0)


class SpeakerMappingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    splits: List[SemanticSplit]
    mappings: List[SpeakerMapping]


def _registry_key(index_name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]", "_", index_name.strip())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        raise ValueError("Conversation index name cannot be empty.")
    digest = hashlib.sha1(index_name.encode("utf-8")).hexdigest()[:8]
    return f"{normalized}_{digest}"


class SpeakerRegistryStore:
    def __init__(self, directory: str):
        self.directory = directory

    def _path(self, index_name: str) -> str:
        return os.path.join(self.directory, f"{_registry_key(index_name)}.json")

    def _lock_path(self, index_name: str) -> str:
        return os.path.join(self.directory, f"{_registry_key(index_name)}.lock")

    @contextmanager
    def lock(self, index_name: str):
        os.makedirs(self.directory, exist_ok=True)
        with open(self._lock_path(index_name), "a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def load(self, index_name: str) -> Dict[str, Any]:
        path = self._path(index_name)
        if not os.path.isfile(path):
            raise ValueError(
                f"Speaker registry for '{index_name}' does not exist. "
                "The first conversation chunk must have chunk_index=0."
            )
        with open(path, encoding="utf-8") as registry_file:
            registry = json.load(registry_file)
        if not isinstance(registry, dict):
            raise ValueError(f"Speaker registry for '{index_name}' is invalid.")
        return registry

    def save(self, index_name: str, registry: Dict[str, Any]) -> None:
        os.makedirs(self.directory, exist_ok=True)
        destination = self._path(index_name)
        fd, temporary_path = tempfile.mkstemp(
            dir=self.directory,
            prefix=f".{_registry_key(index_name)}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as registry_file:
                json.dump(registry, registry_file, ensure_ascii=False, indent=2)
                registry_file.flush()
                os.fsync(registry_file.fileno())
            os.replace(temporary_path, destination)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def delete(self, index_name: str) -> None:
        try:
            os.unlink(self._path(index_name))
        except FileNotFoundError:
            pass


class SpeakerReconciler:
    def prepare(
        self,
        index_name: str,
        chunk_index: int,
        append: bool,
        turns: List[Dict[str, Any]],
        registry: Optional[Dict[str, Any]],
        generate_mapping: Callable[[str], str],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self._validate_chunk(chunk_index, append, turns, registry)

        working_registry = (
            self._new_registry(index_name)
            if chunk_index == 0
            else deepcopy(registry)
        )
        original_local_speakers = self._local_speakers(turns)

        if original_local_speakers:
            registry_before_reconciliation = deepcopy(working_registry)
            try:
                response = self._request_mapping(
                    working_registry,
                    turns,
                    generate_mapping,
                )
                corrected_turns = self._apply_semantic_splits(turns, response.splits)
                local_speakers = self._local_speakers(corrected_turns)
                resolved, proposed_names = self._resolve_mapping(
                    working_registry,
                    local_speakers,
                    response,
                )
                reconciled_turns = self._rewrite_turns(
                    corrected_turns,
                    resolved,
                    proposed_names,
                    working_registry,
                )
            except Exception as exc:
                logger.warning(
                    "Speaker reconciliation failed; preserving the original turns: %s",
                    exc,
                )
                working_registry = registry_before_reconciliation
                resolved = {
                    speaker: self._allocate_speaker(working_registry)
                    for speaker in original_local_speakers
                }
                reconciled_turns = self._rewrite_turns(
                    turns,
                    resolved,
                    {},
                    working_registry,
                )
        else:
            reconciled_turns = deepcopy(turns)
        self._update_registry(working_registry, chunk_index, reconciled_turns)
        return reconciled_turns, working_registry

    @staticmethod
    def _new_registry(index_name: str) -> Dict[str, Any]:
        return {
            "version": 1,
            "conversation_id": index_name,
            "last_chunk_index": -1,
            "next_speaker_number": 1,
            "speakers": {},
            "recent_turns": [],
        }

    @staticmethod
    def _validate_chunk(chunk_index, append, turns, registry) -> None:
        if not isinstance(chunk_index, int) or isinstance(chunk_index, bool) or chunk_index < 0:
            raise ValueError("chunk_index must be a non-negative integer.")
        if not isinstance(turns, list) or not turns:
            raise ValueError("Structured conversation ingestion requires at least one turn.")
        if chunk_index == 0:
            if append:
                raise ValueError("The first conversation chunk must use append=false.")
            return
        if not append:
            raise ValueError("Conversation chunks after chunk 0 must use append=true.")
        if registry is None:
            raise ValueError("A speaker registry is required after the first chunk.")
        expected = registry.get("last_chunk_index", -1) + 1
        if chunk_index != expected:
            raise ValueError(
                f"Out-of-order conversation chunk: expected {expected}, received {chunk_index}."
            )

    @staticmethod
    def _local_speakers(turns: List[Dict[str, Any]]) -> List[str]:
        speakers = []
        for turn in turns:
            if not isinstance(turn, dict):
                raise ValueError("Every transcription turn must be an object.")
            if not str(turn.get("text") or "").strip():
                raise ValueError("Every transcription turn must contain text.")
            raw_speaker = turn.get("speaker")
            speaker = str(raw_speaker).strip() if raw_speaker is not None else ""
            if speaker and speaker not in speakers:
                speakers.append(speaker)
        return speakers

    @staticmethod
    def _allocate_speaker(registry: Dict[str, Any]) -> str:
        speaker_number = int(registry["next_speaker_number"])
        speaker_id = f"Speaker {speaker_number}"
        registry["next_speaker_number"] = speaker_number + 1
        registry["speakers"][speaker_id] = {
            "display_name": None,
            "name_confidence": None,
            "recent_examples": [],
        }
        return speaker_id

    def _request_mapping(
        self,
        registry: Dict[str, Any],
        turns: List[Dict[str, Any]],
        generate_mapping: Callable[[str], str],
    ) -> SpeakerMappingResponse:
        prompt = self._build_prompt(registry, turns)
        raw_response = generate_mapping(prompt)
        if not isinstance(raw_response, str) or not raw_response.strip():
            raise ValueError("Speaker-reconciliation model returned an empty response.")
        try:
            return SpeakerMappingResponse.model_validate_json(raw_response.strip())
        except ValidationError as exc:
            raise ValueError(
                f"Speaker-reconciliation model returned invalid JSON: {exc}"
            ) from exc

    @staticmethod
    def _build_prompt(
        registry: Dict[str, Any],
        turns: List[Dict[str, Any]],
    ) -> str:
        context = {
            "canonical_speakers": registry["speakers"],
            "recent_conversation_turns": registry.get("recent_turns", []),
            "current_chunk_turns": [
                {
                    "turn_index": turn_index,
                    "speaker": turn.get("speaker"),
                    "text": turn.get("text"),
                    "segments": [
                        {
                            "segment_index": segment.get("segment_index"),
                            "speaker": segment.get("speaker"),
                            "text": segment.get("text"),
                        }
                        for segment in turn.get("segments", [])
                        if isinstance(segment, dict)
                    ],
                }
                for turn_index, turn in enumerate(turns)
            ],
        }
        return (
            "First validate the current chunk's merged turns using dialogue semantics. "
            "A turn may be split only between two preserved segments when they clearly "
            "belong to different people, for example a question followed by its answer "
            "or incompatible self-identifications. Do not split quotations, ordinary "
            "multi-sentence speech, topic changes, or ambiguous cases. Emit a split only "
            "when confidence is at least 0.90, and use splits: [] when no such correction "
            "is needed. after_segment_index must exactly copy the segment_index of "
            "the segment before the boundary; do not use its position within the turn. "
            "Each split names the speaker that begins "
            "after the boundary; reuse an existing chunk-local label when appropriate, "
            "otherwise create a label such as 'Corrected 1'. Then map every local label "
            "present after those splits to a conversation-level canonical speaker using "
            "dialogue context, self-identifications, direct address, and reply sequence. "
            "Multiple local labels may map to the same canonical speaker when diarization "
            "has split one person into several labels. "
            "canonical_speaker may only be an exact ID key listed in canonical_speakers, "
            "such as 'Speaker 1'; it must never contain a person's name or a chunk-local "
            "label. If canonical_speakers is empty, every canonical_speaker must be null. "
            "Use an existing canonical_speaker only when supported by context; otherwise "
            "set it to null and set confidence to 0. A mention such as 'Bob called' does "
            "not identify the speaker. "
            "Do not invent names. "
            "Set display_name only for a self-identification or a clearly addressed reply. "
            "For each nullable field, use the JSON value null, not the string 'null'. "
            "Return exactly one compact JSON object with this shape and no markdown: "
            '{"splits":[{"turn_index":0,"after_segment_index":0,'
            '"next_speaker":"Corrected 1","confidence":0.95}],'
            '"mappings":[{"local_speaker":"Speaker 0",'
            '"canonical_speaker":null,"confidence":0.0,'
            '"display_name":null,"name_confidence":null},'
            '{"local_speaker":"Corrected 1","canonical_speaker":null,'
            '"confidence":0.0,"display_name":"Mary","name_confidence":0.98}]}\n'
            f"INPUT={json.dumps(context, ensure_ascii=False)}"
        )

    @staticmethod
    def _apply_semantic_splits(
        turns: List[Dict[str, Any]],
        splits: List[SemanticSplit],
    ) -> List[Dict[str, Any]]:
        splits_by_turn: Dict[int, List[SemanticSplit]] = {}
        for split in splits:
            if split.turn_index >= len(turns):
                raise ValueError(
                    f"Semantic split references unknown turn {split.turn_index}."
                )
            if split.confidence < SEMANTIC_SPLIT_CONFIDENCE:
                raise ValueError(
                    "Semantic splits must have confidence of at least "
                    f"{SEMANTIC_SPLIT_CONFIDENCE:.2f}."
                )
            next_speaker = split.next_speaker.strip()
            if not next_speaker:
                raise ValueError("A semantic split must provide its next speaker.")
            splits_by_turn.setdefault(split.turn_index, []).append(split)

        corrected = []
        for turn_index, source_turn in enumerate(turns):
            turn_splits = splits_by_turn.get(turn_index, [])
            if not turn_splits:
                corrected.append(deepcopy(source_turn))
                continue

            source_speaker = str(source_turn.get("speaker") or "").strip()
            if not source_speaker:
                raise ValueError("A turn without a speaker cannot be semantically split.")

            segments = source_turn.get("segments")
            if not isinstance(segments, list) or len(segments) < 2:
                raise ValueError(
                    f"Turn {turn_index} has no preserved segment boundary to split."
                )

            segment_positions = {}
            for position, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    raise ValueError(
                        f"Turn {turn_index} contains an invalid preserved segment."
                    )
                segment_index = segment.get("segment_index")
                if not isinstance(segment_index, int) or isinstance(segment_index, bool):
                    raise ValueError(
                        f"Turn {turn_index} contains an invalid segment index."
                    )
                if segment_index in segment_positions:
                    raise ValueError(
                        f"Turn {turn_index} contains duplicate segment indices."
                    )
                segment_positions[segment_index] = position

            boundaries = {}
            for split in turn_splits:
                position = segment_positions.get(split.after_segment_index)
                if position is None or position >= len(segments) - 1:
                    logger.warning(
                        "Ignoring invalid semantic split boundary in turn %s: %s.",
                        turn_index,
                        split.after_segment_index,
                    )
                    continue
                if position in boundaries:
                    raise ValueError(
                        f"Duplicate semantic split boundary in turn {turn_index}."
                    )
                boundaries[position] = split.next_speaker.strip()

            active_speaker = source_speaker
            group_start = 0
            for position in sorted(boundaries):
                corrected.append(
                    SpeakerReconciler._turn_from_segments(
                        source_turn,
                        segments[group_start : position + 1],
                        active_speaker,
                    )
                )
                next_speaker = boundaries[position]
                if next_speaker == active_speaker:
                    raise ValueError(
                        f"Semantic split in turn {turn_index} does not change speaker."
                    )
                active_speaker = next_speaker
                group_start = position + 1

            corrected.append(
                SpeakerReconciler._turn_from_segments(
                    source_turn,
                    segments[group_start:],
                    active_speaker,
                )
            )
        return corrected

    @staticmethod
    def _turn_from_segments(
        source_turn: Dict[str, Any],
        segments: List[Dict[str, Any]],
        speaker: str,
    ) -> Dict[str, Any]:
        turn = deepcopy(source_turn)
        turn["speaker"] = speaker
        turn["text"] = " ".join(
            str(segment.get("text") or "").strip()
            for segment in segments
        ).strip()
        if not turn["text"]:
            raise ValueError("A semantic split produced an empty turn.")
        turn["start_time"] = segments[0].get("start_time")
        turn["end_time"] = segments[-1].get("end_time")
        turn["segments"] = deepcopy(segments)
        return turn

    def _resolve_mapping(
        self,
        registry: Dict[str, Any],
        local_speakers: List[str],
        response: SpeakerMappingResponse,
    ) -> Tuple[Dict[str, str], Dict[str, Tuple[str, float]]]:
        by_local = {}
        for mapping in response.mappings:
            if mapping.local_speaker not in local_speakers:
                logger.warning(
                    "Ignoring mapping for unused local speaker '%s'.",
                    mapping.local_speaker,
                )
                continue
            if mapping.local_speaker in by_local:
                raise ValueError(
                    f"Duplicate mapping for local speaker '{mapping.local_speaker}'."
                )
            by_local[mapping.local_speaker] = mapping

        if set(by_local) != set(local_speakers):
            raise ValueError(
                "Speaker mapping must contain every local speaker exactly once."
            )

        resolved = {}
        proposed_names = {}
        for local_speaker in local_speakers:
            mapping = by_local[local_speaker]
            canonical = mapping.canonical_speaker
            if canonical is not None:
                canonical = canonical.strip()
                if canonical not in registry["speakers"]:
                    raise ValueError(
                        f"Unknown canonical speaker '{canonical}' in model response."
                    )

            if canonical is not None and mapping.confidence >= SPEAKER_MATCH_CONFIDENCE:
                speaker_id = canonical
            else:
                speaker_id = self._allocate_speaker(registry)

            display_name = (
                mapping.display_name.strip()
                if isinstance(mapping.display_name, str)
                else ""
            )
            if display_name:
                if mapping.name_confidence is None:
                    raise ValueError(
                        f"Name confidence is required for '{display_name}'."
                    )
                if mapping.name_confidence >= SPEAKER_NAME_CONFIDENCE:
                    proposed_names[speaker_id] = (
                        display_name,
                        mapping.name_confidence,
                    )
            resolved[local_speaker] = speaker_id

        return resolved, proposed_names

    @staticmethod
    def _rewrite_turns(
        turns: List[Dict[str, Any]],
        resolved: Dict[str, str],
        proposed_names: Dict[str, Tuple[str, float]],
        registry: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        reconciled = []
        for source_turn in turns:
            turn = deepcopy(source_turn)
            raw_speaker = turn.get("speaker")
            local_speaker = (
                str(raw_speaker).strip() if raw_speaker is not None else ""
            )
            if not local_speaker:
                turn["local_speaker"] = None
                turn["speaker_id"] = None
                turn["speaker_name"] = None
                turn["speaker"] = None
                reconciled.append(turn)
                continue

            speaker_id = resolved[local_speaker]
            speaker_record = registry["speakers"][speaker_id]
            proposed_name = proposed_names.get(speaker_id)
            if proposed_name is not None:
                display_name, name_confidence = proposed_name
                existing_name = speaker_record.get("display_name")
                if existing_name and existing_name.casefold() != display_name.casefold():
                    raise ValueError(
                        f"Conflicting name for {speaker_id}: existing '{existing_name}', "
                        f"received '{display_name}'."
                    )
                speaker_record["display_name"] = display_name
                speaker_record["name_confidence"] = name_confidence

            display_name = speaker_record.get("display_name")
            turn["local_speaker"] = local_speaker
            turn["speaker_id"] = speaker_id
            turn["speaker_name"] = display_name
            turn["speaker"] = (
                f"{display_name} [{speaker_id}]" if display_name else speaker_id
            )
            reconciled.append(turn)
        return reconciled

    @staticmethod
    def _update_registry(
        registry: Dict[str, Any],
        chunk_index: int,
        turns: List[Dict[str, Any]],
    ) -> None:
        for turn in turns:
            speaker_id = turn.get("speaker_id")
            text = str(turn.get("text") or "").strip()
            if speaker_id:
                examples = registry["speakers"][speaker_id]["recent_examples"]
                examples.append(text)
                del examples[:-MAX_RECENT_EXAMPLES]

        registry["recent_turns"].extend(
            {
                "speaker_id": turn.get("speaker_id"),
                "speaker_name": turn.get("speaker_name"),
                "text": turn.get("text"),
            }
            for turn in turns
        )
        del registry["recent_turns"][:-MAX_RECENT_TURNS]
        registry["last_chunk_index"] = chunk_index
