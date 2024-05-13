#!/bin/env python

import argparse
from pathlib import Path

import numpy as np
from Modules.utils import write_bw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="path to the npz file to convert",
        type=Path,
    )
    args = parser.parse_args()
    with np.load(args.filename) as f:
        signals = {k: f[k] for k in f.keys()}
    write_bw(Path(args.filename.parent, args.filename.stem + ".bw"), signals)
