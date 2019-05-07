#!/bin/bash
mode=$1
low=$2
if [ $# -eq 3 ]
  then
    python cltmi.py $mode $low $3
else
    python cltmi.py $mode $low
fi
