#!/usr/bin/env python

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from Modules import tf_utils
from Modules.tf_utils import correlate, mae_cor


def parsing():
    """
    Parse the command-line arguments.

    Arguments
    ---------
    python command-line

    Returns
    -------
    trained_model : trained model saved by tf.keras.model.save
    dataset : dataset in npz format
    output: Path to the output directory and file name
    """
    # Declaration of expected arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--trained_model",
        help="trained model file, or model weights file. Must be in a format "
        "accepted by tf.keras.models.load_model",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-g",
        "--genome",
        help="one-hot encoded genome file in npz format with an array per "
        "chromosome.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Path to the output directory. If it doesn't exist, it will be created. "
        "However it should be empty otherwise files in it may be overwritten.",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-c",
        "--chromosomes",
        help="chromosomes to predict on. Specify 'all' if you wish to predict "
        "on all chromosomes (default: %(default)s)",
        nargs="+",
        default=["all"],
        type=str,
    )
    parser.add_argument(
        "-s",
        "--strand",
        help="strand to predict on, choose between 'for', 'rev' or "
        "'both' (default: %(default)s)",
        type=str,
        default="both",
    )
    parser.add_argument(
        "-w",
        "--winsize",
        help="Number of base pairs in windows used for training, used only if "
        "winsize can't be inferred from the model",
        type=int,
    )
    parser.add_argument(
        "-h_int",
        "--head_interval",
        help="Spacing between output heads in case of mutliple outputs, "
        "(default: %(default)s)",
        default=None,
        type=int,
    )
    parser.add_argument(
        "-mid",
        "--middle",
        action="store_true",
        help="Indicates to use only predictions in the middle half of the "
        "window, only if head_interval is set",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        help="Number of samples to use for predicting in parallel (default: %(default)s)",
        default=1024,
        type=int,
    )
    args = parser.parse_args()
    # Check if the input data is valid
    if not args.genome.is_file():
        sys.exit(
            f"{args.genome} does not exist.\n" "Please enter valid genome file path."
        )
    return args


if __name__ == "__main__":
    # Get arguments
    args = parsing()
    # Maybe create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    # Limit gpu memory usage
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    # Load trained model
    model = tf.keras.models.load_model(
        args.trained_model, custom_objects={"mae_cor": mae_cor, "correlate": correlate}
    )
    try:
        if isinstance(model.layers[0], tf.keras.layers.InputLayer):
            winsize = model.layers[0].input_shape[0][1]
        else:
            winsize = model.layers[0].input_shape[1]
    except (AttributeError, IndexError):
        print("Couldn't infer winsize from model")
        winsize = args.winsize
    # Initialize predictions
    all_preds = {}
    # Load genome
    with np.load(args.genome) as genome:
        if args.chromosomes == ["all"]:
            chromosomes = genome.keys()
        else:
            chromosomes = args.chromosomes
        for chr_id in chromosomes:
            # Load chromosome one-hot encoded sequence
            try:
                one_hot_chr = genome[chr_id]
            except KeyError:
                warnings.warn(
                    Warning(
                        f"{chr_id} is not a valid chromosome ID in {args.genome}, "
                        "skipping..."
                    )
                )
                continue
            # Predict on specified strands
            if args.strand in ["for", "both"]:
                all_preds[chr_id] = tf_utils.predict(
                    model,
                    one_hot_chr,
                    winsize,
                    batch_size=args.batch_size,
                    head_interval=args.head_interval,
                    middle=args.middle,
                )
            if args.strand in ["rev", "both"]:
                all_preds[f"{chr_id}_rev"] = tf_utils.predict(
                    model,
                    one_hot_chr,
                    winsize,
                    batch_size=args.batch_size,
                    head_interval=args.head_interval,
                    middle=args.middle,
                    reverse=True,
                )
    # Save predicitons
    if args.middle:
        output_file = Path(args.output, f"preds_mid_on_{args.genome.name}")
    else:
        output_file = Path(args.output, f"preds_on_{args.genome.name}")
    np.savez_compressed(output_file, **all_preds)
