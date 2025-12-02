#!/bin/bash
pip install --no-cache-dir -r "requirements.txt"
echo "Installation complete."
python3 "generate_toxic_completions.py"
echo "Run completed"