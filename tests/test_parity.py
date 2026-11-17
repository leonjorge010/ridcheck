"""The exported int8 model matches the C firmware on the baked vectors."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_c_selftest_matches_python_export():
    binary = Path("/tmp/ridcheck_parity")
    source = [
        "gcc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
        "-o", str(binary),
        str(ROOT / "tests" / "parity_main.c"),
        str(ROOT / "firmware" / "src" / "ridcheck_infer.c"),
        "-lm",
    ]
    compiled = subprocess.run(source, cwd=ROOT, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    ran = subprocess.run([str(binary)], cwd=ROOT, capture_output=True, text=True)
    assert ran.returncode == 0, ran.stderr
    assert "selftest ok" in ran.stdout
