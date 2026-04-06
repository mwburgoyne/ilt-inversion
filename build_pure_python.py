#!/usr/bin/env python3
"""
Build a pure-Python wheel (no Rust extension).

Usage:
    python build_pure_python.py

Creates a universal wheel in dist/ that works on any platform.
The Rust acceleration will simply not be available at runtime.
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    dist = ROOT / "dist"
    if dist.exists():
        shutil.rmtree(dist)

    # Temporarily swap pyproject.toml for a setuptools-based one
    pyproject = ROOT / "pyproject.toml"
    pyproject_backup = ROOT / "pyproject.toml.bak"
    pyproject.rename(pyproject_backup)

    # Read version and metadata from the real pyproject.toml so we don't
    # have to maintain them in two places.
    import tomllib
    with open(pyproject_backup, "rb") as f:
        real_meta = tomllib.load(f)["project"]

    pyproject.write_text(f"""\
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "ilt-inversion"
version = "{real_meta['version']}"
description = "{real_meta['description']}"
readme = "README.md"
license = {{text = "GPL-3.0-or-later"}}
requires-python = "{real_meta['requires-python']}"
authors = [{{name = "Mark Burgoyne", email = "mark.burgoyne@gmail.com"}}]
dependencies = ["mpmath>=1.3", "numpy>=1.21"]

[project.optional-dependencies]
fast = ["gmpy2>=2.1"]
flint = ["python-flint>=0.4"]

[tool.setuptools.packages.find]
where = ["python"]
""")

    try:
        subprocess.check_call([
            sys.executable, "-m", "build",
            "--wheel", "--no-isolation",
            str(ROOT),
        ])
    finally:
        pyproject.unlink(missing_ok=True)
        pyproject_backup.rename(pyproject)

    print(f"\nPure-Python wheel built in {dist}/")


if __name__ == "__main__":
    main()
