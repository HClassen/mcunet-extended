#!/bin/bash

if [[ -z ${1} ]]; then
	echo "missing results directory"
	exit 1
fi

echo "CMD [\"python3\", \"main.py\", \"${1}/r1\", \"${1}/r2\"]" >> Dockerfile
# docker build -t hclassen/mcunet:test .
