#!/bin/bash

TARGETDIR="awkres"

mkdir -p "$TARGETDIR"
awk -v targetdir="$TARGETDIR" '
BEGIN{c=1}
/entry/,/Edep/ {
    if ($0 ~ /Edep/) {
	c++
    } else if ($0 !~ /entry/) {
	print > targetdir "/" c
    }
}' $1
