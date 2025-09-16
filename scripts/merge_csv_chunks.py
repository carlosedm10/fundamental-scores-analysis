#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys


def merge_chunks(chunk_dir: Path, output_csv: Path) -> None:
    chunk_dir = Path(chunk_dir)
    output_csv = Path(output_csv)

    if not chunk_dir.exists() or not chunk_dir.is_dir():
        raise FileNotFoundError(f"Chunk directory not found: {chunk_dir}")

    # Collect parts in sorted order
    parts = sorted(
        [p for p in chunk_dir.iterdir() if p.is_file() and p.suffix == ".csv"],
        key=lambda p: p.name,
    )
    if not parts:
        raise FileNotFoundError(f"No .csv chunk files found in: {chunk_dir}")

    # Write header from first chunk, then data rows from all chunks (skip header for subsequent)
    with output_csv.open("wb") as outfile:
        for index, part in enumerate(parts):
            with part.open("rb") as infile:
                if index == 0:
                    # copy everything
                    while True:
                        buf = infile.read(1024 * 1024)
                        if not buf:
                            break
                        outfile.write(buf)
                else:
                    # skip header line, then copy
                    header = infile.readline()
                    # copy remainder
                    while True:
                        buf = infile.read(1024 * 1024)
                        if not buf:
                            break
                        outfile.write(buf)


def main():
    parser = argparse.ArgumentParser(
        description="Merge header-preserving CSV chunks back into a single CSV."
    )
    parser.add_argument(
        "chunk_dir",
        type=Path,
        help="Directory containing partNNN.csv files (e.g., code/data/chunks/Big Data future)",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path for the merged CSV to write (will overwrite if exists)",
    )
    args = parser.parse_args()

    try:
        merge_chunks(args.chunk_dir, args.output_csv)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
