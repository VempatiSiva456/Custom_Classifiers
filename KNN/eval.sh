#!/bin/bash
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <test_file>"
    exit 1
fi

python3 1.py $1