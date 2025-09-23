#!/bin/bash

# Usage: ./extract_last25_to_txt.sh /path/to/directory output.txt

INPUT_DIR="$1"
OUTPUT_FILE="$2"

if [ -z "$INPUT_DIR" ] || [ -z "$OUTPUT_FILE" ]; then
  echo "Usage: $0 /path/to/directory output.txt"
  exit 1
fi

# Clear the output file
> "$OUTPUT_FILE"

# Process each file
find "$INPUT_DIR" -type f | while IFS= read -r FILE; do
  {
    echo "===== $FILE ====="
    tail -n 20 "$FILE"
    echo ""  # Add an empty line between files
  } >> "$OUTPUT_FILE"
done

echo "Done! Output saved to $OUTPUT_FILE"