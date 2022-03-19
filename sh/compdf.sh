#! /bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 DIRECTORY"
    echo "Shrinks all PDF files in DIRECTORY."
    exit 1
else
    folder=$1
fi

for file in $folder/*.pdf; do
    newfile=${file%.pdf}small.pdf   
    gs -dBATCH -dNOPAUSE -sDEVICE=pdfwrite -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -r150 -sOutputFile="$newfile" "$file"
done