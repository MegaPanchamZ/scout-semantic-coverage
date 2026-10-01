"""Cross-route (cross-map) shared obligation suite.

By default each ``policy_search`` arm tracks coverage independently, so the
search on route A never learns what route B already covered. For the suite-
building method that is wrong: the whole point is to accumulate the obligations
covered across routes and keep targeting what is still uncovered.

``SharedSuiteStore`` is a small, lock-protected, atomically-written JSON file
holding the union of covered obligation signatures for one policy arm. Each
process unions the store into its own ``EngineCoverageState.suite_covered`` at
startup and merges its coverage back after every evaluation, so all routes in a
campaign root share one suite (per policy, keeping policy arms comparable).
"""
from __future__ import annotations

import json
import os
import pathlib
import tempfile
from collections.abc import Iterable

try:  # POSIX only; the harness runs on Linux
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


class SharedSuiteStore:
    """Union of covered obligation signatures across routes for one policy."""

    def __init__(self, path: pathlib.Path | str) -> None:
        self.path = pathlib.Path(path)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    # ------------------------------------------------------------------ helpers
    def _locked(self):
        """Context manager holding an exclusive advisory lock on the store."""
        store = self

        class _Ctx:
            def __enter__(self):
                store.path.parent.mkdir(parents=True, exist_ok=True)
                self._handle = open(store._lock_path, "a+")  # noqa: SIM115 - held for the block
                if fcntl is not None:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX)
                return store

            def __exit__(self, *exc):
                if fcntl is not None:
                    fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
                self._handle.close()
                return False

        return _Ctx()

    def _read(self) -> set[str]:
        if not self.path.exists():
            return set()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return set()
        covered = payload.get("covered", []) if isinstance(payload, dict) else payload
        return {str(sig) for sig in covered if isinstance(sig, str)}

    def _write(self, covered: set[str]) -> None:
        payload = {"version": 1, "covered": sorted(covered)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=0)
            os.replace(tmp, self.path)  # atomic on POSIX
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # --------------------------------------------------------------------- api
    def load(self) -> set[str]:
        """Current covered set (safe to call without the caller holding a lock)."""
        with self._locked():
            return self._read()

    def merge(self, signatures: Iterable[str]) -> set[str]:
        """Union ``signatures`` into the store; return the resulting covered set."""
        incoming = {str(sig) for sig in signatures}
        with self._locked():
            covered = self._read() | incoming
            self._write(covered)
            return covered
