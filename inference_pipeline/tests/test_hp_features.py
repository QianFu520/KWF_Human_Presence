import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import hp_features as h  # noqa: E402


def test_lengths():
    assert len(h.HP_FEATURES_25) == 25
    assert len(h.HP_FEATURES_29) == 29


@pytest.mark.parametrize("features", [h.HP_FEATURES_25, h.HP_FEATURES_29])
def test_no_duplicates(features):
    assert len(features) == len(set(features))


def test_25_is_subset_of_29():
    for name in h.HP_FEATURES_25:
        assert name in h.HP_FEATURES_29


def test_select_features_raises_on_missing_column():
    df = pd.DataFrame({name: [0.0] for name in h.HP_FEATURES_25[1:]})
    with pytest.raises(KeyError, match=h.HP_FEATURES_25[0]):
        h.select_features(df, h.HP_FEATURES_25)
