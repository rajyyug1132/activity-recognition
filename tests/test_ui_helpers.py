"""UI helper checks: every activity key a model can emit has a style, and the summary line is honest."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "src"))

from components.ui import act, coalesce, mmss, summarize  # noqa: E402
from inference import ACTS  # noqa: E402

CAMERA_KEYS = ["walking", "standing", "upstairs", "downstairs", "stairs"]


def test_every_model_activity_has_a_style():
    for key in list(ACTS) + CAMERA_KEYS:
        assert act(key).cls != "act-none", key


def test_summary_only_says_mostly_when_it_dominates():
    assert summarize({"WALKING": 7, "SITTING": 3})[0] == "Mostly walking"
    mixed = {"WALKING": 21, "SITTING": 21, "STANDING": 21, "LAYING": 19,
             "WALKING_UPSTAIRS": 17, "WALKING_DOWNSTAIRS": 19}
    assert summarize(mixed)[0] == "A mix of six activities"
    assert summarize({"STANDING": 5})[0] == "All standing"


def test_time_format_and_runs():
    assert mmss(124) == "2:04"
    assert coalesce(["A", "A", "B", "A"]) == [("A", 2), ("B", 1), ("A", 1)]
