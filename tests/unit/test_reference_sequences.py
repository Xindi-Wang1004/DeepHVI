"""Offline acceptance of full-sequence catalogs and resumable public downloads."""

from __future__ import annotations

import csv
import gzip
import io
import json
import sqlite3
from urllib.error import URLError

import pytest

from deephvi.data import reference_download as download
from deephvi.data.reference import fasta_records, prepare_reference
from deephvi.data.reference_download import ReferenceDataError
from deephvi.io import DataFormatError


def fake_fetch(monkeypatch, pages):
    calls = []

    def fetch(url, path):
        calls.append(url)
        value = pages[url]
        if isinstance(value, Exception):
            raise value
        content, headers = value
        path.write_bytes(content)
        return headers

    monkeypatch.setattr(download, "_fetch", fetch)
    return calls


def test_catalog_preserves_all_accessions_full_lengths_and_stable_ids(tmp_path):
    fasta = tmp_path / "input.fasta.gz"
    long_sequence = "ACDE" * 3000
    with gzip.open(fasta, "wt") as handle:
        handle.write(
            ">sp|P12345|FIRST description\nac-d.\ne*\n"
            ">tr|A0A123|SECOND another accession\nACDE\n"
            ">tr|A0A124|LONG full protein\n" + long_sequence + "\n"
        )
    output = tmp_path / "viral"
    manifest = prepare_reference(fasta, output, source="viral")
    assert manifest["statistics"] == {
        "records": 3, "unique_sequences": 2, "accessions": 3,
        "duplicate_sequence_records": 1, "maximum_residues": 12000,
        "records_over_10000_residues": 1,
    }
    with sqlite3.connect(output / "catalog.sqlite3") as db:
        aliases = db.execute("SELECT accession,candidate_id FROM accessions ORDER BY accession").fetchall()
        assert aliases[0][1] == aliases[2][1]
        assert db.execute("SELECT max(length(sequence)) FROM proteins").fetchone() == (12000,)
    with (output / "accessions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert {row["accession"] for row in rows} == {"P12345", "A0A123", "A0A124"}
    assert rows[0]["candidate_id"].startswith("viral_")
    # Reordering the source retains candidate identities and exported gallery bytes.
    reordered = tmp_path / "reordered.fasta"
    reordered.write_text(
        ">tr|A0A124|LONG full protein\n" + long_sequence + "\n"
        ">tr|A0A123|SECOND another accession\nACDE\n"
        ">sp|P12345|FIRST description\nACDE\n"
    )
    other = tmp_path / "other"
    prepare_reference(reordered, other, source="viral")
    assert (output / "reference.fasta").read_bytes() == (other / "reference.fasta").read_bytes()
    original = (output / "manifest.json").read_bytes()
    assert prepare_reference(fasta, output, source="viral") == manifest
    assert (output / "manifest.json").read_bytes() == original
    (output / "reference.fasta").write_text("corruption")
    with pytest.raises(ReferenceDataError, match="fingerprint mismatch"):
        prepare_reference(fasta, output, source="viral")


@pytest.mark.parametrize("text", [
    "", "ACDE\n", ">\nACDE\n", ">NP_1.1\n", ">bad\nACDE\n",
    ">NP_1.1\nAC?E\n", ">NP_1.1\nAC*E\n",
])
def test_invalid_public_fasta_is_rejected(tmp_path, text):
    path = tmp_path / "input.faa"
    path.write_text(text)
    with pytest.raises((ReferenceDataError, DataFormatError)):
        list(fasta_records(path, "human"))


def test_conflicting_accession_is_not_published_and_can_be_rebuilt(tmp_path):
    path = tmp_path / "input.faa"
    path.write_text(">NP_1.1 first\nACDE\n>NP_1.1 conflicting\nAAAA\n")
    output = tmp_path / "human"
    with pytest.raises(ReferenceDataError, match="different sequences"):
        prepare_reference(path, output, source="human")
    assert not (output / "manifest.json").exists()
    path.write_text(">NP_1.1 first\nACDE\n>XP_2.1 second\nACDE\n")
    manifest = prepare_reference(path, output, source="human")
    assert manifest["statistics"]["unique_sequences"] == 1
    assert manifest["statistics"]["accessions"] == 2
    path.write_text(">NP_1.1 changed\nAAAA\n")
    with pytest.raises(ReferenceDataError, match="different inputs"):
        prepare_reference(path, output, source="human")


def test_human_download_verifies_md5_and_completed_receipt(tmp_path, monkeypatch):
    zipped = gzip.compress(b">NP_1.1 fixture\nACDE\n")
    path = tmp_path / "expected.gz"
    path.write_bytes(zipped)
    md5 = download.file_digest(path, "md5")
    checksums = f"{md5}  ./{download.HUMAN_FILE}\n".encode()
    calls = fake_fetch(monkeypatch, {
        download.HUMAN_BASE + "md5checksums.txt": (checksums, {}),
        download.HUMAN_BASE + download.HUMAN_FILE: (zipped, {}),
    })
    output = tmp_path / "raw"
    receipt = download.download_human(output)
    assert receipt["md5"] == md5
    assert receipt["assembly_accession"] == download.ASSEMBLY
    assert len(calls) == 2
    completed = (output / "download.json").read_bytes()
    assert download.download_human(output) == receipt
    assert len(calls) == 2
    assert (output / "download.json").read_bytes() == completed
    manifest = prepare_reference(output / download.HUMAN_FILE, tmp_path / "processed",
                                 source="human", download_receipt=output / "download.json")
    assert manifest["provenance"]["md5"] == md5
    (output / download.HUMAN_FILE).write_bytes(b"corruption")
    with pytest.raises(ReferenceDataError, match="fingerprint mismatch"):
        download.download_human(output)
    with pytest.raises(ReferenceDataError, match="receipt does not match"):
        prepare_reference(output / download.HUMAN_FILE, tmp_path / "other",
                          source="human", download_receipt=output / "download.json")


def test_human_md5_mismatch_never_publishes_receipt(tmp_path, monkeypatch):
    fake_fetch(monkeypatch, {
        download.HUMAN_BASE + "md5checksums.txt": (
            f"{'0' * 32}  ./{download.HUMAN_FILE}\n".encode(), {}),
        download.HUMAN_BASE + download.HUMAN_FILE: (b"bad", {}),
    })
    with pytest.raises(ReferenceDataError, match="MD5 mismatch"):
        download.download_human(tmp_path)
    assert not (tmp_path / "download.json").exists()


def viral_pages():
    next_url = download.UNIPROT_SEARCH + "cursor=isolated_test"
    pages = {
        download.VIRAL_URL: (b">sp|P12345|FIRST\nACDE\n", {
            "x-uniprot-release": "TEST_RELEASE", "x-total-results": "2",
            "link": f'<{next_url}>; rel="next"',
        }),
        next_url: (b">tr|A0A123|SECOND\nACDE\n", {
            "x-uniprot-release": "TEST_RELEASE", "x-total-results": "2",
        }),
    }
    return pages, next_url


def test_viral_download_resumes_pages_and_is_offline_after_completion(tmp_path, monkeypatch):
    pages, next_url = viral_pages()
    second = pages[next_url]
    pages[next_url] = URLError("isolated connection failure")
    calls = fake_fetch(monkeypatch, pages)
    with pytest.raises(URLError):
        download.download_viral(tmp_path)
    checkpoint = json.loads((tmp_path / "download-progress.json").read_text())
    assert checkpoint["records"] == 1
    assert not (tmp_path / "download.json").exists()
    pages[next_url] = second
    receipt = download.download_viral(tmp_path)
    assert calls == [download.VIRAL_URL, next_url, next_url]
    assert receipt["records"] == receipt["expected_records"] == 2
    with gzip.open(tmp_path / download.VIRAL_FILE, "rt") as handle:
        assert handle.read().count(">") == 2
    original = (tmp_path / "download.json").read_bytes()
    assert download.download_viral(tmp_path) == receipt
    assert len(calls) == 3
    assert (tmp_path / "download.json").read_bytes() == original
    manifest = prepare_reference(tmp_path / download.VIRAL_FILE, tmp_path / "processed",
                                 source="viral", download_receipt=tmp_path / "download.json")
    assert manifest["statistics"]["accessions"] == 2
    assert manifest["statistics"]["unique_sequences"] == 1
    assert manifest["provenance"]["uniprot_release"] == "TEST_RELEASE"


@pytest.mark.parametrize("header,value", [
    ("x-uniprot-release", "DIFFERENT_RELEASE"), ("x-total-results", "3"),
])
def test_viral_download_refuses_mixed_versions(tmp_path, monkeypatch, header, value):
    pages, next_url = viral_pages()
    pages[next_url][1][header] = value
    fake_fetch(monkeypatch, pages)
    with pytest.raises(ReferenceDataError, match="refusing mixed versions"):
        download.download_viral(tmp_path)
    assert not (tmp_path / "download.json").exists()
    assert len(json.loads((tmp_path / "download-progress.json").read_text())["pages"]) == 1


def test_viral_resume_checks_saved_page_fingerprint(tmp_path, monkeypatch):
    pages, next_url = viral_pages()
    pages[next_url] = URLError("interrupted")
    calls = fake_fetch(monkeypatch, pages)
    with pytest.raises(URLError):
        download.download_viral(tmp_path)
    (tmp_path / "pages/000000.fasta.gz").write_bytes(b"corruption")
    with pytest.raises(ReferenceDataError, match="page fingerprint mismatch"):
        download.download_viral(tmp_path)
    assert len(calls) == 2


@pytest.mark.parametrize("failure", ["count", "origin", "invalid_fasta"])
def test_viral_invalid_response_is_not_completed(tmp_path, monkeypatch, failure):
    pages, next_url = viral_pages()
    if failure == "count":
        pages[download.VIRAL_URL][1].pop("link")
    elif failure == "origin":
        pages[download.VIRAL_URL][1]["link"] = '<https://example.invalid/>; rel="next"'
    else:
        pages[next_url] = (b"<html>error</html>\n", pages[next_url][1])
    fake_fetch(monkeypatch, pages)
    with pytest.raises(ReferenceDataError):
        download.download_viral(tmp_path)
    assert not (tmp_path / "download.json").exists()


def test_fetch_retries_partial_response_without_appending(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {"Content-Length": "4"}

    responses = iter([Response(b"AC"), Response(b"ACDE")])
    monkeypatch.setattr(download, "_request", lambda url: next(responses))
    monkeypatch.setattr(download.time, "sleep", lambda seconds: None)
    target = tmp_path / "download"
    download._fetch("https://example.invalid/isolated", target)
    assert target.read_bytes() == b"ACDE"
    assert not target.with_name("download.part").exists()


def test_processed_gallery_works_with_existing_retrieval_index(tmp_path):
    # This tests data/CLI compatibility using the explicitly isolated demo encoder.
    from deephvi.data.sequence_backend import HashSequenceBackend
    from deephvi.inference.retrieve import build_index_from_fasta, search

    path = tmp_path / "input.faa"
    path.write_text(">NP_1.1\nACDEFGHIK\n>XP_2.1\nACDEFGHIK\n>XP_3.1\nYYYYYYYYY\n")
    output = tmp_path / "processed"
    prepare_reference(path, output, source="human")
    backend = HashSequenceBackend(dim=32)
    index = build_index_from_fasta(output / "reference.fasta", tmp_path / "index",
                                   config={}, backend=backend)
    assert len(index) == 2
    hits = search(index, backend.encode(["ACDEFGHIK"]), top_k=1)
    assert hits[0][0]["sequence"] == "ACDEFGHIK"
    assert hits[0][0]["id"].startswith("human_")
