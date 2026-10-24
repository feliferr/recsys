import pandas as pd
import pytest

from recsys.evaluation.splits import global_temporal, leave_last_out


def _df() -> pd.DataFrame:
    rows = [
        ("a", "i1", "2024-01-01"),
        ("a", "i2", "2024-01-02"),
        ("a", "i3", "2024-01-03"),
        ("b", "i1", "2024-01-04"),  # single event: stays in train
        ("c", "i2", "2024-01-06"),
        ("c", "i1", "2024-01-05"),  # unsorted on purpose
    ]
    return pd.DataFrame(rows, columns=["user_id", "item_id", "timestamp"]).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"]), rating=None
    )


def test_leave_last_out_uses_latest_event() -> None:
    s = leave_last_out(_df())
    assert dict(zip(s.test["user_id"], s.test["item_id"], strict=True)) == {"a": "i3", "c": "i2"}
    assert "b" in set(s.train["user_id"])
    assert "b" not in set(s.test["user_id"])


def test_leave_last_out_no_future_leakage() -> None:
    s = leave_last_out(_df())
    last_train = s.train.groupby("user_id")["timestamp"].max()
    first_test = s.test.groupby("user_id")["timestamp"].min()
    assert (last_train.loc[first_test.index] < first_test).all()
    assert len(s.train) + len(s.test) == len(_df())


def test_global_temporal_cutoff() -> None:
    s = global_temporal(_df(), test_frac=0.34)
    assert s.train["timestamp"].max() < s.test["timestamp"].min()
    assert len(s.test) > 0


def test_invalid_args() -> None:
    with pytest.raises(ValueError, match="n_test"):
        leave_last_out(_df(), n_test=0)
    with pytest.raises(ValueError, match="test_frac"):
        global_temporal(_df(), test_frac=1.5)
