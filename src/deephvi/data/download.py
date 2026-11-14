"""Archive extraction and dataset discovery.

Archives are untrusted input. Extraction rejects absolute paths, ``..``
components, symlinks and device files, and never runs anything from the
archive. Files are written to an empty destination directory.
"""

from __future__ import annotations

import csv
import shutil
import tarfile
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath, PureWindowsPath

from ..io import DataFormatError
from ..logging_utils import get_logger


class ArchiveError(DataFormatError):
    """Raised when an archive is unsafe or unsupported."""


def _check_member_name(name: str) -> None:
    if not name or name in {".", "./"}:
        return
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        raise ArchiveError(f"Archive member has an absolute path: {name!r}")
    parts = [part for part in posix.parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise ArchiveError(f"Archive member escapes the destination: {name!r}")


def _check_no_symlinks(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArchiveError(
                f"Extracted archive contains a symbolic link: {path}. "
                "The destination is not safe to use; inspect the archive manually."
            )


def extract_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Extract `` archive`` into ``destination`` and return extracted files."""
    archive = Path(archive)
    destination = Path(destination)
    if not archive.exists():
        raise ArchiveError(f"Archive not found: {archive}")
    # Validate before writing: a pre-existing link can redirect extraction
    # outside the destination even when every archive member name is safe.
    if destination.is_symlink() or any(parent.is_symlink() for parent in destination.parents):
        raise ArchiveError(f"Destination must not contain symbolic links: {destination}")
    if destination.exists():
        _check_no_symlinks(destination)
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise ArchiveError(
            f"Destination {destination} is not empty. Choose a new directory or pass --overwrite."
        )
    destination.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive.suffixes).lower()
    if suffixes.endswith(".7z"):
        names = _extract_7z(archive, destination)
    elif suffixes.endswith(".zip"):
        names = _extract_zip(archive, destination)
    elif suffixes.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".tar")):
        names = _extract_tar(archive, destination)
    else:
        raise ArchiveError(
            f"Unsupported archive type: {archive.name}. Supported: .7z, .zip, .tar(.gz|.bz2|.xz)"
        )
    _check_no_symlinks(destination)
    return [destination / name for name in names]


def _extract_7z(archive: Path, destination: Path) -> list[str]:
    try:
        import py7zr
    except ImportError as exc:
        raise ArchiveError(
            "py7zr is required to extract .7z files. Install it with "
            "`uv sync --extra data` or `pip install py7zr`."
        ) from exc
    try:
        with py7zr.SevenZipFile(archive, mode="r") as handle:
            names = handle.getnames()
            for name in names:
                _check_member_name(name)
            for info in handle.files:
                # ArchiveFile exposes link flags on supported py7zr versions;
                # older FileInfo results from list() omit them.
                if (info.is_symlink or getattr(info, "is_junction", False)
                        or getattr(info, "is_socket", False)):
                    raise ArchiveError(f"7z contains a link or special file: {info.filename!r}")
            handle.extractall(path=destination)
    except ArchiveError:
        raise
    except Exception as exc:  # noqa: BLE001 - py7zr raises several exception types
        raise ArchiveError(
            f"Could not extract {archive}: {exc}. If the archive is password-protected or "
            "split across volumes, obtain the complete file."
        ) from exc
    return names


def _extract_zip(archive: Path, destination: Path) -> list[str]:
    names: list[str] = []
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            _check_member_name(info.filename)
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ArchiveError(f"ZIP contains a symbolic link: {info.filename!r}")
            names.append(info.filename)
        handle.extractall(destination)
    return names


def _extract_tar(archive: Path, destination: Path) -> list[str]:
    names: list[str] = []
    with tarfile.open(archive) as handle:
        members = handle.getmembers()
        for member in members:
            _check_member_name(member.name)
            if member.issym() or member.islnk():
                raise ArchiveError(f"tar archive contains a link: {member.name!r}")
            if not (member.isfile() or member.isdir()):
                raise ArchiveError(f"tar archive contains a special file: {member.name!r}")
            names.append(member.name)
        try:
            handle.extractall(destination, filter="data")
        except TypeError:  # Python < 3.12
            handle.extractall(destination)
    return names


def find_pair_tables(root: str | Path) -> list[dict[str, object]]:
    """List CSV files under ``root`` with their sizes, largest first."""
    root = Path(root)
    tables = []
    for path in sorted(root.rglob("*.csv")):
        if "__MACOSX" in path.parts:
            continue
        tables.append({"path": str(path), "size_bytes": path.stat().st_size})
    tables.sort(key=lambda item: int(item["size_bytes"]), reverse=True)
    return tables


def _header_width(path: Path) -> int:
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        line = handle.readline()
    try:
        return len(next(csv.reader([line])))
    except (csv.Error, StopIteration):
        return 0


def repair_truncated_tail(path: str | Path, *, tail_bytes: int = 1 << 20) -> bool:
    """Drop an incomplete final record left by a truncated archive member.

    Some released CSVs end in the middle of a quoted feature vector. Pandas
    (rightly) refuses to parse such a file. A file that does not end in a
    newline is only modified when its last line does not have the same number
    of CSV fields as the header; a complete final record without a trailing
    newline is left untouched.
    """
    path = Path(path)
    size = path.stat().st_size
    if size == 0:
        return False
    with open(path, "rb") as handle:
        handle.seek(max(0, size - tail_bytes))
        chunk = handle.read()
    if chunk.endswith(b"\n"):
        return False
    newline = chunk.rfind(b"\n")
    if newline < 0:
        return False
    last_line = chunk[newline + 1 :].decode("utf-8", errors="replace")
    expected = _header_width(path)
    if not expected:
        return False
    try:
        fields = next(csv.reader([last_line], strict=True))
    except (csv.Error, StopIteration):
        fields = []
    if len(fields) == expected:
        return False
    new_size = max(0, size - tail_bytes) + newline + 1
    with open(path, "r+b") as handle:
        handle.truncate(new_size)
    get_logger().warning(
        "Dropped an incomplete trailing record from %s (%d bytes); the archive member is truncated.",
        path,
        size - new_size,
    )
    return True


def prepare_dataset(
    archive: str | Path,
    destination: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, object]:
    extracted = extract_archive(archive, destination, overwrite=overwrite)
    repaired = [
        str(path)
        for path in find_extracted_files(destination, suffixes=(".csv",))
        if repair_truncated_tail(path)
    ]
    tables = find_pair_tables(destination)
    return {
        "archive": str(archive),
        "destination": str(destination),
        "n_extracted": len(extracted),
        "repaired_files": repaired,
        "tables": tables,
    }


def find_extracted_files(root: str | Path, suffixes: Iterable[str] = (".csv", ".tsv")) -> list[Path]:
    root = Path(root)
    wanted = tuple(suffix.lower() for suffix in suffixes)
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in wanted
    )


def disk_free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(Path(path))
    return round(usage.free / 1e9, 2)


def esm_download_instructions() -> str:
    return (
        "Place ESM-2 files in data/models/esm2, or download "
        "facebook/esm2_t6_8M_UR50D from Hugging Face "
        "and set data.esm.model to its local path. The demo backend `hash` needs no weights."
    )
