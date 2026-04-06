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

    # Write a temporary setup.py for pure-Python build
    setup_py = ROOT / "setup.py"
    setup_py.write_text("""\
from setuptools import setup, find_packages

setup(
    name="ilt-inversion",
    version="0.1.0",
    description="Fast numerical inverse Laplace transforms: GWR + Fixed Talbot",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Mark Burgoyne",
    author_email="mark.burgoyne@gmail.com",
    license="GPL-3.0-or-later",
    python_requires=">=3.10",
    install_requires=["mpmath>=1.3", "numpy>=1.21"],
    extras_require={
        "fast": ["gmpy2>=2.1"],
        "flint": ["python-flint>=0.4"],
    },
    packages=find_packages(where="python"),
    package_dir={"": "python"},
)
""")

    try:
        subprocess.check_call([
            sys.executable, "-m", "build",
            "--wheel", "--no-isolation",
            str(ROOT),
        ])
    finally:
        setup_py.unlink(missing_ok=True)

    print(f"\\nPure-Python wheel built in {dist}/")


if __name__ == "__main__":
    main()
