from __future__ import annotations

from pathlib import Path

from deephvi.data.download import repair_truncated_tail


def test_truncated_final_record_is_dropped(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        'S_H,S_V,feature_SH,feature_SV,label\n'
        'ACDE,FGHI,"[1, 2]","[3, 4]",1\n'
        'ACDF,FGHI,"[5, 6]","[7, 8]",0\n'
        'ACDG,FGHI,"[9, 10]","[11,',
        encoding="utf-8",
    )
    assert repair_truncated_tail(path) is True
    text = path.read_text(encoding="utf-8")
    assert text.endswith("0\n")
    assert "ACDG" not in text


def test_complete_record_without_trailing_newline_is_kept(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    path.write_text('S_H,S_V,label\nACDE,FGHI,1', encoding="utf-8")
    assert repair_truncated_tail(path) is False
    assert path.read_text(encoding="utf-8").endswith("ACDE,FGHI,1")


def test_complete_file_is_untouched(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    original = "S_H,S_V,label\nACDE,FGHI,1\n"
    path.write_text(original, encoding="utf-8")
    assert repair_truncated_tail(path) is False
    assert path.read_text(encoding="utf-8") == original
