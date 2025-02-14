#!/bin/bash

while : ; do
        (
                python main.py -c 8 1000 1000 ./results
        ) &

        sleep $((60 * 90))
        killall python

        for f in $(find ~/mcunet-extended/results -name "*.csv" -type f); do
                wc -l ${f}
        done
done
