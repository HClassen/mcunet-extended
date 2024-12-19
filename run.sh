#!/bin/bash

if [[ -z ${1} ]]; then
	echo "missing results directory"
	exit 1
fi

docker run -d --gpus all -v ${1}:/results hclassen/mcunet:test
