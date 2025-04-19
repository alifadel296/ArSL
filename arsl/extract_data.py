import argparse
import os
import zipfile

import py7zr


def extract_archive(args):
    # Ensure destination folder exists
    os.makedirs(args.extract_to, exist_ok=True)

    for file_path in args.file_paths:
        # Determine archive type and extract accordingly
        if file_path.lower().endswith(".zip"):
            
            with zipfile.ZipFile(file_path, "r") as archive:
                archive.extractall(args.extract_to)
            print(f"Extracted ZIP: {file_path} -> {args.extract_to}")
            
        elif file_path.lower().endswith(".7z"):
            
            with py7zr.SevenZipFile(file_path, mode="r") as archive:
                archive.extractall(path=args.extract_to)
            print(f"Extracted 7Z: {file_path} -> {args.extract_to}")
            
        else:
            
            raise ValueError("Unsupported file format. Only ZIP and 7Z files are supported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract the zip and 7z archive of data")
    
    parser.add_argument(
        "-fp",
        "--file_paths",
        type=str,
        nargs="*",
        required=True,
        help="the zip/7z archive path",
    )
    
    parser.add_argument(
        "-et",
        "--extract_to",
        type=str,
        required=True,
        help="The path of folder where you will extract the archives",
    )
    
    args = parser.parse_args()
    extract_archive(args)
