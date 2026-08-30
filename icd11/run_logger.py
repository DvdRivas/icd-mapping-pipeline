"""
run_logger.py
-------------
Session log for the 10-run experiment: everything printed to stdout is also
appended to a single log file, split by a header that names each run.

Why a single file
=================
The experiment is one session of 10 runs over the same 201 diagnoses, and
what matters when auditing it is comparing run N against run M. Ten separate
files would force stitching them back together; one file with explicit
headers keeps the whole session readable top to bottom and diffable.

The file is opened in APPEND mode, so re-running the experiment never
destroys the previous session. Each session starts with a banner recording
the configuration that produced it — model, host, sampling settings,
catalogue version, source files — which is what makes a run reproducible
after the fact.

tqdm progress bars are deliberately left out: they write to stderr, so the
log keeps the substance (statistics, warnings, per-row incidents) without
the carriage-return noise of a progress bar.

Usage
=====
    with RunLogger(path) as log:
        log.session_header({"model": ..., "temperature": 0})
        for i, csv_path in enumerate(csv_paths, 1):
            log.run_header(i, len(csv_paths), csv_path)
            ...                      # every print() lands in the log too
"""

import sys
from datetime import datetime
from pathlib import Path


DEFAULT_LOG_NAME = "experiment-runs.log"


class _Tee:
    """Write-through proxy: everything goes to the console and to the file."""

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, data):
        self._stream.write(data)
        self._handle.write(data)
        return len(data)

    def flush(self):
        self._stream.flush()
        self._handle.flush()

    def isatty(self):
        # tqdm asks; answer for the real console so bars keep working there.
        return self._stream.isatty()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class RunLogger:
    """Tee stdout into a session log, with a header per run."""

    def __init__(self, path: str | Path | None = None, enabled: bool = True):
        self.path    = Path(path) if path else Path(__file__).parent / DEFAULT_LOG_NAME
        self.enabled = enabled
        self._handle = None
        self._saved  = None

    # ── context manager ──────────────────────────────────────────────────

    def __enter__(self) -> "RunLogger":
        if self.enabled:
            self._handle = open(self.path, "a", encoding="utf-8")
            self._saved  = sys.stdout
            sys.stdout   = _Tee(self._saved, self._handle)
        return self

    def __exit__(self, *exc):
        if self._handle:
            sys.stdout = self._saved
            self._handle.write(
                f"\n[session closed {datetime.now():%Y-%m-%d %H:%M:%S}]\n\n"
            )
            self._handle.close()
            self._handle = None
        return False

    # ── headers ──────────────────────────────────────────────────────────

    def session_header(self, config: dict) -> None:
        """
        Banner opening the session, recording what produced these results.

        Printed through stdout so it shows on the console as well: the reader
        should see the exact configuration before any run output.
        """
        if not self.enabled:
            return
        print()
        print("#" * 78)
        print(f"# SESSION START  {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("#" + "-" * 77)
        width = max((len(k) for k in config), default=0)
        for key, value in config.items():
            print(f"#   {key:<{width}} : {value}")
        print("#" * 78)
        print()

    def run_header(self, index: int, total: int, csv_path: str) -> None:
        """Header separating one run from the next inside the same file."""
        print()
        print("=" * 78)
        print(f"= RUN {index}/{total}  ·  {Path(csv_path).name}")
        print(f"= started {datetime.now():%Y-%m-%d %H:%M:%S}")
        print("=" * 78)

    def note(self, text: str) -> None:
        print(f"[log] {text}")
