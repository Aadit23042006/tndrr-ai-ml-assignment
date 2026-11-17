#!/usr/bin/env bash
# Offline tests of the LOGIC (no GPU / internet / real model needed). They do NOT replace your real runs.
set -e
cd "$(dirname "$0")"
pip install -q numpy pypdf faiss-cpu reportlab torch transformers peft tokenizers 2>/dev/null || true
python tests/test_task1.py
python tests/test_task2.py
python tests/test_task3.py
python tests/test_report.py
echo "ALL OFFLINE TESTS PASSED"
