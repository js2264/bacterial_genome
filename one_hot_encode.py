#!/bin/env python

import argparse
from pathlib import Path

import numpy as np
from Modules.utils import one_hot_encode, read_fasta

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="path to the fasta file to convert",
        type=Path,
    )
    args = parser.parse_args()
    genome = read_fasta(args.filename)
    one_hot_genome = one_hot_encode(args.filename)
    np.savez(Path(args.filename.parent, args.filename.stem + ".npz"), **one_hot_genome)
