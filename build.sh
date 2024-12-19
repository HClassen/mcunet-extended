#!/bin/bash

if [[ -z ${1} ]]; then
	echo "missing results directory"
	exit 1
fi

echo "CMD [\"python3\", \"main.py\", \"/gtsrb\", \"${1}\"]" >> Dockerfile
