#!/usr/bin/env python3
"""Copy dataset and randomly downsample to match the smallest class count."""

import os
import random
import shutil

SRC = "./dataset-replaced"
DST = "./dataset-matched"


def main():
    # 1. Remove destination if exists, then copy entire source tree
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    print(f"Copied {SRC} -> {DST}")

    # 2. Count files per subfolder in the copied destination
    subdirs = [
        os.path.join(DST, d)
        for d in sorted(os.listdir(DST))
        if os.path.isdir(os.path.join(DST, d))
    ]

    counts = {}
    for sd in subdirs:
        files = [
            f for f in os.listdir(sd)
            if os.path.isfile(os.path.join(sd, f))
        ]
        counts[sd] = files
        print(f"  {os.path.basename(sd)}: {len(files)} files")

    # 3. Find the minimum count
    min_count = min(len(v) for v in counts.values())
    print(f"\nTarget count per class: {min_count}")

    # 4. Randomly delete excess files
    for sd, files in counts.items():
        excess = len(files) - min_count
        if excess > 0:
            to_delete = random.sample(files, excess)
            for fname in to_delete:
                os.remove(os.path.join(sd, fname))
            print(f"  Removed {excess} files from {os.path.basename(sd)}")
        else:
            print(f"  {os.path.basename(sd)} already at target count")

    print("\nDone. Final counts:")
    for sd in sorted(counts):
        name = os.path.basename(sd)
        n = len([
            f for f in os.listdir(sd)
            if os.path.isfile(os.path.join(sd, f))
        ])
        print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
