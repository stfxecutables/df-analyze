import os
import sys
import threading
from contextlib import AbstractContextManager
from pprint import pformat


class Debug:
    """Printing mixin a la https://doc.rust-lang.org/std/fmt/trait.Debug.html"""

    def __repr__(self) -> str:
        return "".join(pformat(self.__dict__, indent=2, width=80, compact=False))


def capture_outputs() -> None:
    os.set_inheritable(1, True)  # ensure stdout is inherited
    os.set_inheritable(2, True)  # ensure stderr is inherited

    os.dup2(w_fd, 1)
    if merge_stderr:
        os.dup2(w_fd, 2)
    os.set_inheritable(1, True)
    os.set_inheritable(2, True)


class Tee(AbstractContextManager):
    """
    Implemented based on various sources:

    IPython OS-level stdout/stderr capture #1230
    --------------------------------------------
    https://github.com/ipython/ipython/issues/1230, which cites:
    https://github.com/ipython/ipython/commit/97262e9521f384fe8f55566bdfa1b4794e55bd8b

    capture-and-print-subprocess-output.py
    --------------------------------------
    NOTE: below does not work on Windows
    https://gist.github.com/nawatts/e2cdca610463200c12eac2a14efc0bfb

    https://lucadrf.dev/blog/python-subprocess-buffers/  # useful info






    """

    def __init__(self, logfile: str, merge_stderr: bool = True, bufsize: int = 65536):
        self.logfile = logfile
        self.merge_stderr = merge_stderr
        self.bufsize = bufsize

    def __enter__(self):
        self._log = open(self.logfile, "ab", buffering=0)
        r, w = os.pipe()
        self._r, self._w = r, w

        self._out_fd = sys.stdout.fileno()
        self._err_fd = sys.stderr.fileno()

        # Keep the real terminal fds so we can still display to screen
        self._out_save = os.dup(self._out_fd)
        self._err_save = os.dup(self._err_fd)

        # Make the pipe write end inheritable so child processes also write into it
        os.set_inheritable(self._w, True)

        # Redirect stdout (and optionally stderr) into the pipe
        os.dup2(self._w, self._out_fd)
        if self.merge_stderr:
            os.dup2(self._w, self._err_fd)

        # Child processes inherit std handles on all OSes under multiprocessing
        os.set_inheritable(self._out_fd, True)
        os.set_inheritable(self._err_fd, True)

        os.close(self._w)  # keep writers only via fd 1/2
        self._w = None

        # Pump: read from pipe, write to terminal and log
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()

        # Ensure we restore even on abrupt exits
        atexit.register(self._restore)
        return self

    def _pump(self):
        with (
            os.fdopen(self._r, "rb", closefd=True) as r,
            os.fdopen(self._out_save, "wb", closefd=False) as tty,
        ):
            while True:
                data = r.read(self.bufsize)
                if not data:
                    break
                try:
                    tty.write(data)
                    tty.flush()
                except Exception:
                    pass
                try:
                    self._log.write(data)
                    self._log.flush()
                except Exception:
                    pass

    def _restore(self):
        try:
            os.dup2(self._out_save, self._out_fd)
            os.dup2(self._err_save, self._err_fd)
        except Exception:
            pass

    def __exit__(self, exc_type, exc, tb):
        self._restore()
        try:
            self._log.flush()
        finally:
            self._log.close()
        return False
