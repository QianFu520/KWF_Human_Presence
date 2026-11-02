"""
Tests for the validation scripts: combine_reports.py on two small fake summary.json files, and
which reference clips validate_features.py expects a run to produce. No audio.

Run with: python -m pytest inference_pipeline/tests -v
"""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "validation"))

import combine_reports  # noqa: E402
from hp_features import ACOUSTIC  # noqa: E402
from validate_features import expected_reference_clips  # noqa: E402


def fake_summary(folder, recorder, day, expected, matched, worst):
    """summary.json as validate_features.py writes it; `worst` = (feature, max relative difference)."""
    features = {f: {"median_rel_diff": 1e-7, "max_rel_diff": 2e-6, "nan": 0, "match": True} for f in ACOUSTIC}
    feature, value = worst
    features[feature] = {"median_rel_diff": 1e-5, "max_rel_diff": value, "nan": 0, "match": value < 1e-3}
    return {
        "raw_folders": [folder], "recorders": [recorder], "dates": [day],
        "generated": "2026-10-24 12:00", "repo_commit": "abc1234",
        "reference_file": "test_acoustic_features.csv", "reference_clips_total": 2183,
        "clips_produced": 28800, "clips_wrong_length_or_rate": 0,
        "reference_clips_expected": expected, "reference_clips_matched": matched, "clips_compared": matched,
        "match_tolerance": 1e-3, "features": features,
    }


@pytest.fixture
def report_parent(tmp_path):
    """report_am4 passes; report_am6 misses one reference clip and Spectral_Contrast is over 1e-3."""
    parent = tmp_path / "kwf_val"
    for name, summary in [
        ("report_am4", fake_summary("Audio_Moth_4_032025", "Audio_Moth_4", "2025-03-20", 178, 178, ("MFCC_13", 1.73e-4))),
        ("report_am6", fake_summary("Audio_Moth_6_032025", "Audio_Moth_6", "2025-03-20", 171, 170, ("Spectral_Contrast", 2e-3))),
    ]:
        (parent / name).mkdir(parents=True)
        (parent / name / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return parent


def test_combined_table_and_totals(report_parent):
    table, totals = combine_reports.combine(
        combine_reports.load_summaries(combine_reports.find_summaries([report_parent]))
    )
    rows = table.splitlines()[2:]
    assert rows[0] == "| Audio_Moth_4 | 2025-03-20 | 28,800 | 178 / 178 | 1.73e-04 (MFCC_13) | pass |"
    assert rows[1] == (
        "| Audio_Moth_6 | 2025-03-20 | 28,800 | 170 / 171 | 2.00e-03 (Spectral_Contrast) | "
        "FAIL: 1 of 171 reference clips not matched; features differ: Spectral_Contrast |"
    )
    assert totals[0] == ("Matched 348 of 2,183 reference clips in test_acoustic_features.csv across 2 recorders; "
                         "349 were expected for the recorders and dates tested.")
    assert totals[1].startswith("1 of 2 runs pass; 57,600 clips checked, 0 with")
    assert totals[2] == "Largest difference overall: 2.00e-03 (Spectral_Contrast, Audio_Moth_6)."


def test_cli_takes_output_folders_and_writes_markdown(report_parent, tmp_path):
    out = tmp_path / "combined_report.md"
    combine_reports.main([str(report_parent / "report_am6"), str(report_parent / "report_am4"), "--output", str(out)])
    text = out.read_text(encoding="utf-8")
    assert text.index("| Audio_Moth_4 |") < text.index("| Audio_Moth_6 |")  # sorted by recorder
    assert "Matched 348 of 2,183 reference clips" in text


def test_same_recorder_day_twice_raises(report_parent):
    paths = combine_reports.find_summaries([report_parent / "report_am4", report_parent])
    with pytest.raises(SystemExit, match="Audio_Moth_4_032025"):
        combine_reports.load_summaries(paths)


def test_expected_reference_clips_use_folder_date():
    # Audio_Moth_4_032025 records into 2025-03-21 06:25, but the reference clips later that day
    # come from the next folder; other recorders and days are not expected either
    names = ["Audio_Moth_4_20250320_163019", "Audio_Moth_4_20250321_081821",
             "Audio_Moth_6_20250320_155656", "Audio_Moth_4_20250319_082921"]
    folder_days = [("Audio_Moth_4_032025", "Audio_Moth_4", date(2025, 3, 20))]
    assert expected_reference_clips(names, folder_days) == ["Audio_Moth_4_20250320_163019"]
