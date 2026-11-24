import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from inference import smooth

mk = lambda a: pd.DataFrame({"start_time": range(len(a)), "activity": a, "confidence": 1.0})
src = mk(list("WWSWWW"))
assert list(smooth(src, 5)["activity"]) == list("WWWWWW")      # flicker removed
assert list(src["activity"]) == list("WWSWWW")                  # input untouched
assert list(smooth(mk(list("AABB")), 3)["activity"]) == list("AABB")  # real change kept
assert smooth(mk(list("AB")), 1)["activity"].tolist() == ["A", "B"]  # k=1 is identity
print("ok")
