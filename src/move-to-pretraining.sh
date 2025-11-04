#!/bin/bash
pushd "$(dirname "$0")"
# Directory containing pretraining directories
source_dir="./data"

# Target directory for training
target_dir="./data/pretraining"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Loop through all pretraining_* directories
for dir in "$source_dir"/pretraining_*; do
    if [ -d "$dir" ]; then
        echo "Processing directory: $dir"
        for file in "$dir"/*; do
            if [ -f "$file" ]; then
                # Get the file extension
                extension="${file##*.}"
                # Generate a unique name with UUID and the extension
                unique_name="$(uuidgen | cut -d'-' -f1).$extension"
                target_file="$target_dir/$unique_name"

                # Copy the file
                cp "$file" "$target_file"
                echo "Copied $file to $target_file"
            fi
        done
    fi
done

echo "All files have been copied to $target_dir with unique names based on UUIDs and extensions."
popd
