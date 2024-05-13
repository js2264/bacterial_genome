#!/bin/env python

import argparse
from pathlib import Path

import numpy as np
from Modules.utils import load_bw

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="path to the bigwig file to convert",
        type=Path,
    )
    args = parser.parse_args()
    signals = load_bw(args.filename)
    np.savez(Path(args.filename.parent, args.filename.stem + ".npz"), **signals)
