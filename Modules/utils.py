#!/usr/bin/env python

from typing import Iterable

import numpy as np
import pyBigWig
from numpy.lib.stride_tricks import as_strided


def load_bw(filename, nantonum=True):
    labels = {}
    with pyBigWig.open(str(filename)) as bw:
        for chr_id in bw.chroms():
            if nantonum:
                labels[chr_id] = np.nan_to_num(bw.values(chr_id, 0, -1, numpy=True))
            else:
                labels[chr_id] = bw.values(chr_id, 0, -1, numpy=True)
    return labels


def write_bw(filename, signals):
    bw = pyBigWig.open(str(filename), "w")
    bw.addHeader([(k, len(v)) for k, v in signals.items()])
    for chr_id, val in signals.items():
        bw.addEntries(chr_id, 0, values=val, span=1, step=1)
    bw.close()


def merge_chroms(chr_ids: Iterable[str], file: str):
    """Concatenate chromosomes by interspacing them with a value of 0.

    Parameters
    ----------
    chr_ids: iterable of str
        Names of the chromosomes to concatenate, names must match keys in `file`
    file: str
        Name of an npz file containing one-hot encoded chromosomes.
        The length of the chromosome must be the first dimension.

    Returns
    -------
    ndarray
        Concatenated one-hot encoded array.
    """
    annot = []
    with np.load(file) as f:
        for chr_id in chr_ids:
            annot.append(f[chr_id])
            shape, dtype = f[chr_id].shape, f[chr_id].dtype
            annot.append(np.zeros((1,) + shape[1:], dtype=dtype))
    return np.concatenate(annot)


def read_fasta(file):
    """Parse a fasta file as a dictionary."""
    with open(file) as f:
        genome = {}
        seq, seqname = "", ""
        for line in f:
            if line.startswith(">"):
                if seqname != "" or seq != "":
                    genome[seqname] = seq
                seqname = line[1:].rstrip()
                seq = ""
            else:
                seq += line.rstrip()
        if seq != "":
            genome[seqname] = seq
    return genome


def one_hot_encode(seq, length=None, one_hot_type=bool, order="ACGT"):
    if length is None:
        length = len(seq)
    one_hot = np.zeros((length, 4), dtype=one_hot_type)
    for i, base in enumerate(seq):
        if i >= length:
            break
        if base.upper() == order[0]:
            one_hot[i, 0] = 1
        elif base.upper() == order[1]:
            one_hot[i, 1] = 1
        elif base.upper() == order[2]:
            one_hot[i, 2] = 1
        elif base.upper() == order[3]:
            one_hot[i, 3] = 1
    return one_hot


def RC_one_hot(one_hot, order):
    """Compute reverse complement of one_hot array.

    Parameters
    ----------
    one_hot : ndarray, shape=(n, 4)
        Array of one-hot encoded DNA, with one_hot values along last axis
    order : str, default='ACGT'
        String representation of the order in which to encode bases. Default
        value of 'ACGT' means that A has the representation with 1 in first
        position, C with 1 in second position, etc...

    Returns
    -------
    ndarray
        Reverse complement of one_hot.
    """
    # Dictionary mapping base to its complement
    base_to_comp = dict(zip("ACGT", "TGCA"))
    # Array to reorder one_hot columns
    converter = np.zeros(4, dtype=int)
    for i, c in enumerate(order):
        converter[order.find(base_to_comp[c])] = i
    return one_hot[::-1, converter]


def strided_sliding_window_view(
    x, window_shape, stride, sliding_len, axis=None, *, subok=False, writeable=False
):
    """Return blocks of sliding windows, with starts seperated by a stride.

    Variant of `sliding_window_view` from the numpy library, use with caution as it is not as robust.

    This will provide blocks of sliding window of `sliding_len` windows,
    with first windows spaced by `stride`
    The axis parameter determines where the stride and slide are performed, it
    can only be a single value.

    Examples
    --------
    >>> strided_sliding_window_view(np.arange(10), 3, 4, 2)
    array([[[0, 1, 2],
            [1, 2, 3]],

           [[4, 5, 6],
            [5, 6, 7]]])
    >>> strided_sliding_window_view(np.arange(10), 3, 2, 4)
    array([[[0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5]],

           [[2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7]],

           [[4, 5, 6],
            [5, 6, 7],
            [6, 7, 8],
            [7, 8, 9]]])
    """
    window_shape = tuple(window_shape) if np.iterable(window_shape) else (window_shape,)
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError("`window_shape` cannot contain negative values")

    # ADDED THIS ####
    stride = tuple(stride) if np.iterable(stride) else (stride,)
    stride_array = np.array(stride)
    if np.any(stride_array < 0):
        raise ValueError("`stride` cannot contain negative values")
    if len(stride) == 1:
        stride += (1,)
    elif len(stride) > 2:
        raise ValueError("`stride` cannot be of length greater than 2")
    if sliding_len % stride[1] != 0:
        raise ValueError("second `stride` must divide `sliding_len` exactly")
    # CHANGED THIS ####
    # if axis is None:
    #     axis = tuple(range(x.ndim))
    #     if len(window_shape) != len(axis):
    #         raise ValueError(f'Since axis is `None`, must provide '
    #                          f'window_shape for all dimensions of `x`; '
    #                          f'got {len(window_shape)} window_shape '
    #                          f'elements and `x.ndim` is {x.ndim}.')
    # else:
    #     axis = normalize_axis_tuple(axis, x.ndim, allow_duplicate=True)
    #     if len(window_shape) != len(axis):
    #         raise ValueError(f'Must provide matching length window_shape '
    #                          f'and axis; got {len(window_shape)} '
    #                          f'window_shape elements and {len(axis)} axes '
    #                          f'elements.')
    # TO ###################
    if axis is None:
        axis = 0
    ########################

    # CHANGED THIS LINE ####
    # out_strides = ((x.strides[0]*stride, )
    #                + tuple(x.strides[1:])
    #                + tuple(x.strides[ax] for ax in axis))
    # TO ###################
    out_strides = (
        x.strides[:axis]
        + (x.strides[axis] * stride[0], x.strides[axis] * stride[1])
        + x.strides[axis:]
    )
    ########################

    # CHANGED THIS ####
    # note: same axis can be windowed repeatedly
    # x_shape_trimmed = list(x.shape)
    # for ax, dim in zip(axis, window_shape):
    #     if x_shape_trimmed[ax] < dim:
    #         raise ValueError(
    #             'window shape cannot be larger than input array shape')
    #     x_shape_trimmed[ax] = int(np.ceil(
    #         (x_shape_trimmed[ax] - dim + 1) / stride))
    # out_shape = tuple(x_shape_trimmed) + window_shape
    # TO ###################
    x_shape_trimmed = [
        (x.shape[axis] - window_shape[axis] - sliding_len + stride[1]) // stride[0] + 1,
        sliding_len // stride[1],
    ]
    out_shape = window_shape[:axis] + tuple(x_shape_trimmed) + window_shape[axis:]
    ########################
    return as_strided(
        x, strides=out_strides, shape=out_shape, subok=subok, writeable=writeable
    )
