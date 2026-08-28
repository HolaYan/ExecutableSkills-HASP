"""Minimal local sandbox for PF evidence code (small-case enumeration).

Runs a short python snippet in a separate interpreter with a wall-clock
timeout, an rlimit on CPU/memory, no network (env scrubbed, sockets
unavailable through the pre-exec shim), and stdout captured. This is for
OFFLINE evidence generation on the cluster; it is not a security boundary
against a hostile author — the code comes from our own judge model on our
own prompts.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import textwrap

_PRELUDE = textwrap.dedent('''
    import resource, sys, builtins
    resource.setrlimit(resource.RLIMIT_CPU, (8, 8))
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    import socket
    def _no(*a, **k): raise OSError("network disabled")
    socket.socket = _no
    import itertools, math, functools, collections
    from itertools import combinations, permutations, product, combinations_with_replacement
    from math import comb, perm, factorial, gcd
''')


def run_python(code: str, timeout_s: float = 10.0) -> tuple[bool, str]:
    """Returns (ok, stdout_or_error). ok=False on timeout / non-zero exit."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(_PRELUDE + "\n" + code)
        path = f.name
    try:
        env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "HOME": tempfile.gettempdir()}
        p = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True,
                           timeout=timeout_s, env=env)
        if p.returncode != 0:
            return False, (p.stderr or "")[-800:]
        return True, p.stdout[-4000:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
