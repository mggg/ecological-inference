#! /bin/bash

set -ex # fail on first error, print commands

SRC_DIR=${SRC_DIR:-$(pwd)}

echo "Checking code style with black..."
python -m black --line-length 100 --check "${SRC_DIR}"
echo "Success!"

echo "Type checking with mypy..."
mypy --ignore-missing-imports pyei
echo "Success!"

echo "Checking code style with pylint..."
python -m pylint "${SRC_DIR}"/pyei/ "${SRC_DIR}"/test/*.py
echo "Success!"
