import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts" / "materialize_gate2.py"
SPEC = importlib.util.spec_from_file_location("materialize_gate2", PATH)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_event_schema_has_no_embedding_and_all_required_fields():
    assert MOD.EVENT_SCHEMA.names == [
        "uid", "timestamp", "group_id", "feedback_type", "item_id",
        "sid_1", "sid_2", "sid_3", "sid_4", "split",
    ]
    assert "audio_embedding" not in MOD.EVENT_SCHEMA.names


def test_catalog_has_embedding_once():
    assert MOD.CATALOG_SCHEMA.names[-1] == "audio_embedding"
    assert MOD.CATALOG_SCHEMA.field("audio_embedding").type.list_size == 128


def test_group_ids_are_shared_without_tie_breaking():
    times = np.asarray([10, 10, 20, 20, 20, 40], dtype=np.uint32)
    unique, counts = np.unique(times, return_counts=True)
    gids = 100 + np.searchsorted(unique, times)
    assert gids.tolist() == [100, 100, 101, 101, 101, 102]
    assert counts.tolist() == [2, 3, 1]
