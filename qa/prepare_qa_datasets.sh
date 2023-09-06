#!/bin/bash

# Create tab-separated file for question answering.
# It needs to have the following columns:
# quesion,context,answer

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <questions> <contexts> <answers> <output>" >&2
  exit 1
fi

questions=$1
contexts=$2
answers=$3
output=$4

echo -e "question\tcontext\tanswer" > "$output"
paste "$questions" "$contexts" "$answers" \
  | grep -v 'No Answer$' \
  | grep -Pv '\t$' \
  | grep -Pv '\t\t' \
  | grep -Pv '^\t' \
	 >> "$output"
