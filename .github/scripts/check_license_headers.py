# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

#!/usr/bin/env python3

import argparse
import sys

REQUIRED_LICENSE_TEXT = "Copyright (c) Meta Platforms, Inc. and affiliates."


def check_license_header(file_path):
    """Check if a Python file has the required license header."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            return REQUIRED_LICENSE_TEXT in content
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Check license headers in Python files")
    parser.add_argument("files", nargs="*", help="Files to check")
    args = parser.parse_args()

    if not args.files:
        return 0

    missing_headers = []

    for file_path in args.files:
        if file_path.endswith(".py"):
            if not check_license_header(file_path):
                missing_headers.append(file_path)

    if missing_headers:
        print("Missing license headers in the following files:")
        for file_path in missing_headers:
            print(f"  - {file_path}")
        print("\nPlease add the following license header to the top of each file:")
        print("# Copyright (c) Meta Platforms, Inc. and affiliates.")
        print("# All rights reserved.")
        print("#")
        print("# This source code is licensed under the BSD 3-Clause license found in the")
        print("# LICENSE file in the root directory of this source tree.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
