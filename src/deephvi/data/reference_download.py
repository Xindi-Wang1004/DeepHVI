"""Version-checked public reference downloads (no model or HTTP dependencies)."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ASSEMBLY = "GCF_000001405.40"
HUMAN_FILE = f"{ASSEMBLY}_GRCh38.p14_protein.faa.gz"
HUMAN_BASE = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/"
    f"{ASSEMBLY}_GRCh38.p14/"
)
VIRAL_FILE = "uniprotkb_taxonomy_id_10239.fasta.gz"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search?"
VIRAL_URL = UNIPROT_SEARCH + urlencode({
    "query": "taxonomy_id:10239", "format": "fasta", "size": 500, "sort": "accession asc",
})


class ReferenceDataError(RuntimeError):
    """A reference source or a persisted fingerprint is inconsistent."""


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


@contextmanager
def directory_lock(output: Path):
    """The lock file persists; flock itself is released even after a crash."""
    import fcntl

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".reference.lock").open("a+b") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReferenceDataError(f"Another reference command is using {output}") from error
        try:
            yield
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)


def _request(url: str):
    return urlopen(Request(url, headers={"User-Agent": "DeepHVI-reference/1"}), timeout=90)


def _fetch(url: str, path: Path, *, attempts: int = 6) -> dict[str, str]:
    """Retry an individual file/page; never append partial HTTP responses."""
    temporary = path.with_name(path.name + ".part")
    for attempt in range(attempts):
        try:
            with _request(url) as response, temporary.open("wb") as handle:
                headers = {key.lower(): value for key, value in response.headers.items()}
                shutil.copyfileobj(response, handle)
                length = headers.get("content-length")
                if length is not None and handle.tell() != int(length):
                    raise ReferenceDataError("Incomplete HTTP response")
            temporary.replace(path)
            return headers
        except (OSError, URLError, HTTPException, ReferenceDataError):
            if attempt == attempts - 1:
                raise
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def _completed(output: Path, source: str) -> dict | None:
    path = output / "download.json"
    if not path.exists():
        return None
    receipt = json.loads(path.read_text(encoding="utf-8"))
    filename = HUMAN_FILE if source == "human" else VIRAL_FILE
    target = output / filename
    if (receipt.get("download_version") != 1 or receipt.get("source") != source
            or receipt.get("status") != "completed" or receipt.get("file") != filename
            or not target.is_file() or receipt.get("bytes") != target.stat().st_size
            or receipt.get("sha256") != file_digest(target)):
        raise ReferenceDataError("Completed reference download fingerprint mismatch")
    return receipt


def _receipt(target: Path, source: str, **metadata) -> dict:
    return {
        "download_version": 1, "status": "completed", "source": source,
        "file": target.name, "bytes": target.stat().st_size, "sha256": file_digest(target),
        "finished_at": timestamp(), **metadata,
    }


def download_human(output: Path) -> dict:
    """Download the complete assembly protein FASTA and verify NCBI's MD5."""
    output = Path(output)
    with directory_lock(output):
        receipt = _completed(output, "human")
        if receipt is not None:
            return receipt
        checksums = output / "md5checksums.txt"
        # Keep the same checksum table across retries, pinning the annotation bytes.
        if not checksums.exists():
            _fetch(HUMAN_BASE + checksums.name, checksums)
        matches = [line.split()[0] for line in checksums.read_text().splitlines()
                   if len(line.split()) == 2 and line.split()[1] == "./" + HUMAN_FILE]
        if len(matches) != 1 or not re.fullmatch(r"[0-9a-fA-F]{32}", matches[0]):
            raise ReferenceDataError("NCBI checksum table lacks an unambiguous protein FASTA MD5")
        expected = matches[0].lower()
        target = output / HUMAN_FILE
        if not target.exists() or file_digest(target, "md5") != expected:
            _fetch(HUMAN_BASE + HUMAN_FILE, target)
        if file_digest(target, "md5") != expected:
            raise ReferenceDataError("NCBI protein FASTA MD5 mismatch; use a new output directory if annotation changed")
        # Consume the gzip footer as well; a compressed file must be complete.
        with gzip.open(target, "rb") as handle:
            for _ in iter(lambda: handle.read(1024 * 1024), b""):
                pass
        receipt = _receipt(
            target, "human", url=HUMAN_BASE + HUMAN_FILE, assembly_accession=ASSEMBLY,
            md5=expected, checksum_url=HUMAN_BASE + checksums.name,
            checksum_sha256=file_digest(checksums),
            scope="Complete RefSeq assembly protein FASTA, including isoforms, patches and alternate loci.",
        )
        atomic_json(output / "download.json", receipt)
        return receipt


def _page_count(path: Path) -> int:
    with path.open("rb") as handle:
        first = handle.readline()
        if not first.startswith(b">"):
            raise ReferenceDataError("UniProt page is not FASTA")
        count, last = 1, first
        for line in handle:
            count += line.startswith(b">")
            last = line
        if not last.endswith(b"\n") or not 1 <= count <= 500:
            raise ReferenceDataError("Incomplete or oversized UniProt FASTA page")
    return count


def download_viral(output: Path) -> dict:
    """Resume pinned UniProt pages; refuse to mix releases or result counts."""
    output = Path(output)
    with directory_lock(output):
        receipt = _completed(output, "viral")
        if receipt is not None:
            return receipt
        progress = output / "download-progress.json"
        state = json.loads(progress.read_text()) if progress.exists() else {
            "initial_url": VIRAL_URL, "next_url": VIRAL_URL, "pages": [], "records": 0,
            "started_at": timestamp(),
        }
        if state.get("initial_url") != VIRAL_URL:
            raise ReferenceDataError("UniProt query differs from saved progress")
        records = 0
        for number, page in enumerate(state["pages"]):
            if page["file"] != f"pages/{number:06d}.fasta.gz":
                raise ReferenceDataError("Invalid saved UniProt page path")
            path = output / page["file"]
            if not path.is_file() or file_digest(path) != page["sha256"]:
                raise ReferenceDataError("Saved UniProt page fingerprint mismatch")
            records += page["records"]
        if records != state["records"]:
            raise ReferenceDataError("Saved UniProt record count mismatch")
        (output / "pages").mkdir(exist_ok=True)
        while state["next_url"]:
            url = state["next_url"]
            if not url.startswith(UNIPROT_SEARCH) or url in {p["url"] for p in state["pages"]}:
                raise ReferenceDataError("Unexpected or repeated UniProt page URL")
            raw = output / "page.fasta.part"
            headers = _fetch(url, raw)
            count = _page_count(raw)
            release = headers.get("x-uniprot-release")
            total = int(headers.get("x-total-results", "0"))
            if not release or total <= 0:
                raise ReferenceDataError("UniProt release or total-results header is missing")
            if not state["pages"]:
                state.update(uniprot_release=release, expected_records=total,
                             uniprot_release_date=headers.get("x-uniprot-release-date"))
            elif release != state["uniprot_release"] or total != state["expected_records"]:
                raise ReferenceDataError("UniProt release changed; refusing mixed versions. Use a new output directory")
            links = headers.get("link", "")
            match = re.search(r'<([^>]+)>;\s*rel="next"', links)
            next_url = match.group(1) if match else None
            if (next_url and not next_url.startswith(UNIPROT_SEARCH)
                    or state["records"] + count > total):
                raise ReferenceDataError("Unexpected UniProt pagination or record count")
            filename = f"pages/{len(state['pages']):06d}.fasta.gz"
            with (output / filename).open("wb") as handle:
                with gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as zipped:
                    with raw.open("rb") as source:
                        shutil.copyfileobj(source, zipped)
            raw.unlink()
            state["pages"].append({"file": filename, "url": url, "records": count,
                                   "sha256": file_digest(output / filename)})
            state.update(next_url=next_url, records=state["records"] + count, updated_at=timestamp())
            atomic_json(progress, state)
            if len(state["pages"]) % 20 == 0 or not next_url:
                print(json.dumps({"pages": len(state["pages"]), "records": state["records"],
                                  "expected_records": total, "uniprot_release": release}),
                      file=sys.stderr, flush=True)
        if state["records"] != state["expected_records"]:
            raise ReferenceDataError("Downloaded count differs from UniProt total-results")
        target = output / VIRAL_FILE
        temporary = target.with_name(target.name + ".part")
        with temporary.open("wb") as handle:
            for page in state["pages"]:
                with (output / page["file"]).open("rb") as source:
                    shutil.copyfileobj(source, handle)
        temporary.replace(target)  # Concatenated gzip members are a valid gzip FASTA.
        receipt = _receipt(
            target, "viral", url=VIRAL_URL, query="taxonomy_id:10239", **state,
            scope="All matching UniProtKB entries, reviewed and unreviewed, including descendant taxa.",
        )
        atomic_json(output / "download.json", receipt)
        return receipt
