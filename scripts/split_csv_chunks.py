#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


def split_csv(
    input_path: Path,
    target_chunk_bytes: int = 1900 * 1024 * 1024,
    suffix_template: str = ".part{index:03d}.csv",
):
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("rb") as infile:
        header = infile.readline()
        if not header:
            return []

        chunk_index = 0
        bytes_in_chunk = 0
        out_files = []
        out_file = None

        def open_new_chunk(idx: int):
            nonlocal out_file, bytes_in_chunk
            if out_file:
                out_file.close()
            chunk_name = input_path.with_name(
                input_path.name + suffix_template.format(index=idx)
            )
            # Avoid overwriting existing chunks
            if chunk_name.exists():
                chunk_name.unlink()
            out = chunk_name.open("wb")
            out.write(header)
            bytes_written = len(header)
            out_files.append(chunk_name)
            bytes_in_chunk = bytes_written
            return out

        out_file = open_new_chunk(chunk_index)

        for line in infile:
            # If adding this line would exceed target chunk size, roll to next chunk
            if bytes_in_chunk + len(
                line
            ) > target_chunk_bytes and bytes_in_chunk > len(header):
                chunk_index += 1
                out_file = open_new_chunk(chunk_index)
            out_file.write(line)
            bytes_in_chunk += len(line)

        if out_file:
            out_file.close()

    return out_files


def main():
    parser = argparse.ArgumentParser(
        description="Split a CSV into ~size-limited chunks, preserving the header in each chunk."
    )
    parser.add_argument("csv", type=Path, help="Path to the CSV file to split")
    parser.add_argument(
        "--size-mb",
        type=int,
        default=1900,
        help="Target chunk size in MB (default: 1900)",
    )
    args = parser.parse_args()

    target_bytes = args.size_mb * 1024 * 1024
    out_files = split_csv(args.csv, target_chunk_bytes=target_bytes)
    for p in out_files:
        print(p)


if __name__ == "__main__":
    main()
