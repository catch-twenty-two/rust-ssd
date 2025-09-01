#!/usr/bin/env bash
set -e

FOLDER_URL="https://drive.google.com/drive/folders/1H-eqIdc-C6X6T7Q2xPqmWd4Pe-lmh-Kz?usp=drive_link"
OUTPUT_DIR="./../assets/pretrained_models"

# Move to the directory where this script is located
cd "$(dirname "$0")"

# If the folder already exists, skip downloading
if [ -d "$OUTPUT_DIR" ]; then
    echo "Directory '$OUTPUT_DIR' already exists. Skipping..."
    exit 0
fi

# Ensure gdown is installed system-wide
if ! command -v gdown &> /dev/null; then
    echo "gdown not found, (https://pypi.org/project/gdown/)"
    echo "You can install it with:"
    echo "pip install gdown"
    echo "Exiting.."
    exit 1
fi

# Download the folder recursively
echo "Downloading Google Drive folder into '$OUTPUT_DIR'..."
gdown --folder "$FOLDER_URL" -O "$OUTPUT_DIR"

echo "Download complete. Files saved in '$OUTPUT_DIR/'"