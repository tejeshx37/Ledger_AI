"""File checksumming, used to verify dataset integrity and to fingerprint
inputs recorded in a run manifest.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1024 * 1024


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of a file's contents.

    Reads in fixed-size chunks so large dataset files do not need to be
    loaded into memory at once.
    """
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()
