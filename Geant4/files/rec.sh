#!/bin/bash

################################################################################
#
# It takes as argument a file named 'file.1' of the following form:
#
# var1 value1
# var2 value2
# ...
# varN valueN
#
# and assuming that in the same directory, there exist files named 'file.2',
#'file.3', ..., 'file.N', and that the variable names var1, var2, ..., varN,
# are the same across the various files, it reconstructs a series of:
#
#	var1(value1, value2, ..., valueN)
#	var2(value1, value2, ..., valueN)
#	...
#	varN(value1, value2, ..., valueN)
#
# and saves them in the files named 'var1', 'var2', ..., 'varN', in the same
# working directory.
#
################################################################################

fname="${1%.1}"	# remove the .1 part of the filename

for i in "$fname".? "$fname".??;
do
    allfiles="$allfiles $i"
done

echo "-> Assuming '$fname' is the base filename"
echo "  -> We are going to operate on [ $allfiles ]"

awk -F' ' '{ print $1 }' $1 |
while IFS= read -r var;
do
    if [[ -z "$var" ]]; then
	continue
    fi

    echo "-> Reconstructing series for variable '$var'"

    # Reconstruct the series
    grep "$var" $allfiles | awk -F '[\t ]+' '{ print $3 }' > "$var"

    # Check whether resultant files have the same size (they ought to)
    size=$(wc -l "$var" | awk -F ' ' '{ print $1}')
    if [[ -n $prevsize ]]; then		# if $prevsize is already set
	if [[ $size != $prevsize ]]; then
	    echo "Size mismatch: size=$size, prevsize=$prevsize"
	    exit 1
	fi
    else
	prevsize="$size"
    fi
done
