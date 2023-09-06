#!/bin/bash

# Creates a CSV file, two columns, first column is concatenation of question
# and context (in this order), second column is true if answer is in context,
# false otherwise

if [ "$#" -ne 4 ]; then
  echo "Usage: $0 <questions> <contexts> <answers> <output>" >&2
  exit 1
fi

questions=$1
contexts=$2
answers=$3
output=$4


# first column: paste -d" " "$questions" "$contexts"
# second column will be false iff text in answers exactly matches "No Answer":

echo -e "text\tlabel" > "$output"
paste <(paste -d" " "$questions" "$contexts") \
      <(awk '{if ($0 == "No Answer") {print "0"} else {print "1"}}' "$answers") \
      >> "$output"
