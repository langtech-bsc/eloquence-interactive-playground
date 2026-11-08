import json
import tempfile
import unittest

from gradio_app.backend.speaker_reconciliation import (
    SpeakerReconciler,
    SpeakerRegistryStore,
)


def segment(index, speaker, text, start, end):
    return {
        "segment_index": index,
        "speaker": speaker,
        "text": text,
        "start_time": start,
        "end_time": end,
    }


def turn(speaker, components):
    return {
        "speaker": speaker,
        "text": " ".join(item["text"] for item in components),
        "start_time": components[0]["start_time"],
        "end_time": components[-1]["end_time"],
        "segments": components,
    }


def mapping(local, canonical=None, confidence=0.0, name=None, name_confidence=None):
    return {
        "local_speaker": local,
        "canonical_speaker": canonical,
        "confidence": confidence,
        "display_name": name,
        "name_confidence": name_confidence,
    }


class SpeakerReconcilerTests(unittest.TestCase):
    def setUp(self):
        self.reconciler = SpeakerReconciler()

    def prepare_first_chunk(self):
        turns = [
            turn(
                "Speaker 0",
                [segment(0, "Speaker 0", "My name is Alice.", 0.0, 1.0)],
            ),
            turn(
                "Speaker 1",
                [segment(1, "Speaker 1", "Nice to meet you.", 1.1, 2.0)],
            ),
        ]
        return self.reconciler.prepare(
            index_name="conversation",
            chunk_index=0,
            append=False,
            turns=turns,
            registry=None,
            generate_mapping=lambda _: json.dumps({
                "splits": [],
                "mappings": [
                    mapping("Speaker 0", name="Alice", name_confidence=0.98),
                    mapping("Speaker 1"),
                ],
            }),
        )

    def test_first_chunk_uses_llm_and_assigns_names(self):
        reconciled, registry = self.prepare_first_chunk()

        self.assertEqual(reconciled[0]["speaker"], "Alice [Speaker 1]")
        self.assertEqual(reconciled[1]["speaker"], "Speaker 2")
        self.assertEqual(registry["last_chunk_index"], 0)

    def test_semantic_split_creates_two_real_turns(self):
        merged = turn(
            "Speaker 0",
            [
                segment(0, "Speaker 0", "Hi! How are you?", 0.0, 1.0),
                segment(1, "Speaker 0", "Hi! I am Mary.", 1.1, 2.0),
            ],
        )

        reconciled, _ = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=0,
            append=False,
            turns=[merged],
            registry=None,
            generate_mapping=lambda _: json.dumps({
                "splits": [{
                    "turn_index": 0,
                    "after_segment_index": 0,
                    "next_speaker": "Corrected 1",
                    "confidence": 0.97,
                }],
                "mappings": [
                    mapping("Speaker 0"),
                    mapping(
                        "Corrected 1",
                        name="Mary",
                        name_confidence=0.99,
                    ),
                ],
            }),
        )

        self.assertEqual(len(reconciled), 2)
        self.assertEqual(reconciled[0]["text"], "Hi! How are you?")
        self.assertEqual(reconciled[0]["speaker_id"], "Speaker 1")
        self.assertEqual(reconciled[1]["text"], "Hi! I am Mary.")
        self.assertEqual(reconciled[1]["speaker"], "Mary [Speaker 2]")
        self.assertEqual(reconciled[1]["start_time"], 1.1)

    def test_invalid_semantic_split_falls_back_to_original_turn(self):
        merged = turn(
            "Speaker 0",
            [
                segment(0, "Speaker 0", "Question?", 0.0, 1.0),
                segment(1, "Speaker 0", "Answer.", 1.1, 2.0),
            ],
        )

        reconciled, registry = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=0,
            append=False,
            turns=[merged],
            registry=None,
            generate_mapping=lambda _: json.dumps({
                "splits": [{
                    "turn_index": 0,
                    "after_segment_index": 0,
                    "next_speaker": "Corrected 1",
                    "confidence": 0.70,
                }],
                "mappings": [mapping("Speaker 0"), mapping("Corrected 1")],
            }),
        )

        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["text"], "Question? Answer.")
        self.assertEqual(registry["last_chunk_index"], 0)

    def test_ignores_split_outside_preserved_boundary(self):
        merged = turn(
            "Speaker 0",
            [
                segment(4, "Speaker 0", "Question?", 0.0, 1.0),
                segment(5, "Speaker 0", "Answer.", 1.1, 2.0),
            ],
        )

        reconciled, _ = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=0,
            append=False,
            turns=[merged],
            registry=None,
            generate_mapping=lambda _: json.dumps({
                "splits": [{
                    "turn_index": 0,
                    "after_segment_index": 0,
                    "next_speaker": "Corrected 1",
                    "confidence": 0.99,
                }],
                "mappings": [mapping("Speaker 0"), mapping("Corrected 1")],
            }),
        )

        self.assertEqual(len(reconciled), 1)
        self.assertEqual(reconciled[0]["text"], "Question? Answer.")

    def test_following_chunk_reuses_canonical_speakers(self):
        _, registry = self.prepare_first_chunk()
        current = turn(
            "Speaker 0",
            [segment(2, "Speaker 0", "The keys are here.", 30.0, 31.0)],
        )

        reconciled, updated = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=1,
            append=True,
            turns=[current],
            registry=registry,
            generate_mapping=lambda _: json.dumps({
                "splits": [],
                "mappings": [mapping("Speaker 0", "Speaker 1", 0.96)],
            }),
        )

        self.assertEqual(reconciled[0]["speaker"], "Alice [Speaker 1]")
        self.assertEqual(updated["next_speaker_number"], 3)

    def test_low_confidence_match_creates_new_speaker(self):
        _, registry = self.prepare_first_chunk()
        current = turn(
            "Speaker 0",
            [segment(2, "Speaker 0", "I am not sure.", 30.0, 31.0)],
        )

        reconciled, updated = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=1,
            append=True,
            turns=[current],
            registry=registry,
            generate_mapping=lambda _: json.dumps({
                "splits": [],
                "mappings": [mapping("Speaker 0", "Speaker 1", 0.50)],
            }),
        )

        self.assertEqual(reconciled[0]["speaker_id"], "Speaker 3")
        self.assertIn("Speaker 3", updated["speakers"])

    def test_multiple_local_speakers_can_share_canonical_speaker(self):
        _, registry = self.prepare_first_chunk()
        current = [
            turn("Speaker 0", [segment(2, "Speaker 0", "First.", 30.0, 31.0)]),
            turn("Speaker 2", [segment(3, "Speaker 2", "Second.", 31.1, 32.0)]),
        ]

        reconciled, _ = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=1,
            append=True,
            turns=current,
            registry=registry,
            generate_mapping=lambda _: json.dumps({
                "splits": [],
                "mappings": [
                    mapping("Speaker 0", "Speaker 1", 0.96),
                    mapping("Speaker 2", "Speaker 1", 0.94),
                ],
            }),
        )

        self.assertEqual(
            [item["speaker_id"] for item in reconciled],
            ["Speaker 1", "Speaker 1"],
        )

    def test_invalid_model_response_falls_back_to_original_turns(self):
        reconciled, registry = self.reconciler.prepare(
            index_name="conversation",
            chunk_index=0,
            append=False,
            turns=[turn(
                "Speaker 0",
                [segment(0, "Speaker 0", "Hello.", 0.0, 1.0)],
            )],
            registry=None,
            generate_mapping=lambda _: "```json\n{}\n```",
        )

        self.assertEqual(reconciled[0]["text"], "Hello.")
        self.assertEqual(reconciled[0]["speaker_id"], "Speaker 1")
        self.assertEqual(registry["last_chunk_index"], 0)

    def test_rejects_out_of_order_chunk(self):
        _, registry = self.prepare_first_chunk()

        with self.assertRaisesRegex(ValueError, "expected 1, received 2"):
            self.reconciler.prepare(
                index_name="conversation",
                chunk_index=2,
                append=True,
                turns=[turn(
                    "Speaker 0",
                    [segment(2, "Speaker 0", "Hello.", 60.0, 61.0)],
                )],
                registry=registry,
                generate_mapping=lambda _: "{}",
            )

    def test_registry_store_round_trip_and_delete(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SpeakerRegistryStore(directory)
            _, registry = self.prepare_first_chunk()

            with store.lock("conversation/name"):
                store.save("conversation/name", registry)
                self.assertEqual(
                    store.load("conversation/name")["last_chunk_index"],
                    0,
                )
                store.delete("conversation/name")

            with self.assertRaisesRegex(ValueError, "does not exist"):
                store.load("conversation/name")


if __name__ == "__main__":
    unittest.main()
