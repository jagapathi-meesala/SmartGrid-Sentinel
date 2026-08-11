#!/usr/bin/env bash
# =============================================================================
# preprocessing/download_hai.sh
# Downloads the HAI 21.03 dataset (test1..test5.csv) directly from GitHub.
# No license request form needed, unlike CIC-IDS-2017.
#
# Run: bash preprocessing/download_hai.sh
# =============================================================================
set -e
TARGET_DIR="$(dirname "$0")/../dataset/hai-21.03"
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

echo "Downloading HAI 21.03 test files (real labeled attack windows)..."
for f in test1 test2 test3 test4 test5; do
    if [ -f "${f}.csv" ]; then
        echo "  ${f}.csv already present, skipping"
        continue
    fi
    echo "  Fetching ${f}.csv.gz ..."
    curl -sL "https://raw.githubusercontent.com/icsdataset/hai/master/hai-21.03/${f}.csv.gz" -o "${f}.csv.gz"
    gunzip "${f}.csv.gz"
    echo "  -> ${f}.csv ($(wc -l < ${f}.csv) rows)"
done

echo ""
echo "Done. Files are in ${TARGET_DIR}/"
echo "Next: python preprocessing/preprocess_hai.py"
