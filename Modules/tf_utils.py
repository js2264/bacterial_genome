#!/usr/bin/env python
import keras.backend as K
import numpy as np
from Modules import utils
from tensorflow.keras.utils import Sequence


class WindowGenerator(Sequence):
    """
    Build a Generator for training Tensorflow model from chromosome windows.

    Parameters
    ----------
    data : ndarray, shape=(n, 4)
        2D-array of one-hot encoded chromosome.
    labels : ndarray, shape=(n,)
        array of labels for each base of the chromosome.
    winsize : int
        length of windows to send as input to the model
    batch_size : int
        number of windows to send per batch
    max_data : int
        maximum number of windows per epoch (before evaluating on the
        validation set). If there are more windows, they will be used in a
        further epoch. There may be multiple epochs without reusing data.
    shuffle : bool, default=True
        If True, indicates to shuffle windows before training and once all
        windows have been used (not necessarily after each epoch).
    same_samples : bool, default=False
        If True, indicates to use the same sample at each epoch. Otherwise the
        sample is changed at each epoch to use all available data.
    balance : {None, "batch", "global"}, default=None
        "batch" indicates to balance weights among classes inside each batch,
        while "global" indicates to balance on the entire data. Default value
        None indicates not to balance weights.
    n_classes : int, default=500
        If `balance` is set, indicates number of bins to divide the signal
        range in, for determining classes.
    strand : {'for', 'rev', 'both'}, default='both'
        Indicates which strand to use for training. If 'both', half the
        windows of each batch are reversed.
    extradims : int or tuple of int, default=None
        Extra dimensions with length of 1 needed for model inputs
    head_interval : int, default=None
        For multiple outputs accross the entire window, specifies spacing
        between each head. head will start on the far left of the window.
    remove_indices : ndarray, default=None
        1D-array of indices of labels to remove from training.
    remove0s : bool, default=True
        Specifies to remove all labels equal to 0 from training.
    removeNs : bool, default=True
        Specifies to remove all windows containing Ns from training.
    seed : int, default=None
        Seed to use for random shuffles

    Attributes
    ----------
    data : ndarray, shape=(n, 4)
        same as in Parameters
    labels : ndarray, shape=(n, 1)
        same as in Parameters, but with dimension expanded
    winsize : int
        same as in Parameters
    batch_size : int
        same as in Parameters
    max_data : int
        same as in Parameters. But `max_data` is set to the number of windows
        if `max_data` is larger.
    shuffle : bool
        same as in Parameters
    same_samples : bool, default=False
        same as in Parameters
    balance : {None, "batch", "global"}, default=None
        same as in Parameters
    n_classes : int, default=500
        same as in Parameters
    strand : {'for', 'rev', 'both'}, default='for'
        same as in Parameters
    extradims : int or tuple of int
        sam as in Parameters
    head_interval : int, default=None
        same as in Parameters
    remove_indices : ndarray, default=None
        same as in Parameters
    remove0s : bool, default=True
        same as in Parameters
    indexes : ndarray
        1D-array of valid window indexes for training. Valid windows exclude
        windows with Ns or null labels. Indexescorrespond to the center of the
        window in `data`.
    masked_labels : MaskedArray
        1D-array of labels to use for training, invalid labels are masked.
        This complements `indexes` for multiple heads, when a window must be
        kept but some heads must be discarded
    sample : ndarray
        1D-array of indexes to use for the current epoch
    weights : ndarray
        global weights to use when `balance` is set to 'global'.
    start_idx : int
        Index of the starting window to use for next epoch when `same_samples`
        is set to False. It is used and updated in the method `on_epoch_end`.
    """

    def __init__(
        self,
        data,
        labels,
        winsize,
        batch_size,
        max_data,
        shuffle=True,
        same_samples=False,
        balance=None,
        n_classes=500,
        strand="both",
        extradims=None,
        head_interval=None,
        remove_indices=None,
        remove0s=False,
        removeNs=False,
        seed=None,
    ):
        self.data = data
        self.labels = labels
        self.winsize = winsize
        self.batch_size = batch_size
        self.max_data = max_data
        self.shuffle = shuffle
        self.same_samples = same_samples
        self.balance = balance
        self.n_classes = n_classes
        self.strand = strand
        self.extradims = extradims
        self.head_interval = head_interval
        self.remove_indices = remove_indices
        self.remove0s = remove0s

        try:
            assert 0 <= np.min(self.labels)
            assert np.max(self.labels) <= 1
            assert np.allclose(np.min(self.labels), 0)
            assert np.allclose(np.max(self.labels), 1)
        except AssertionError:
            print("labels must be normalized between 0 and 1")
            raise

        # Select indices of windows to train on, using masked arrays
        # Some window indices are totally removed from the training set, but
        # with multiple heads, some individual labels can be removed from one
        # of the heads without throwing away the window. This is done by
        # weighting these labels to 0.
        # `self.indexes` stores windows used for training
        # `self.masked_labels` is a masked array with all invalid labels masked
        self.indexes = np.ma.arange(len(self.data))
        self.masked_labels = np.ma.array(self.labels, mask=False)
        # Remove indices of edge windows
        edge_window_mask = (self.indexes < self.winsize // 2) | (
            self.indexes >= len(self.data) - ((self.winsize - 1) // 2)
        )
        if not self.head_interval:
            # With multiple heads, even edge labels can be predicted
            self.masked_labels[edge_window_mask] = np.ma.masked
        self.indexes[edge_window_mask] = np.ma.masked
        if removeNs:
            # Remove windows containing at least one N
            N_mask = np.sum(self.data, axis=1) == 0
            N_window_mask = np.asarray(
                np.convolve(N_mask, np.ones(self.winsize), mode="same"), dtype=int
            )
            self.masked_labels[N_window_mask] = np.ma.masked
            self.indexes[N_window_mask] = np.ma.masked
        if self.remove0s:
            self.masked_labels[self.labels == 0] = np.ma.masked
            if not self.head_interval:
                self.indexes[self.labels == 0] = np.ma.masked
        if self.remove_indices is not None:
            self.masked_labels[self.remove_indices] = np.ma.masked
            if not self.head_interval:
                self.indexes[self.remove_indices] = np.ma.masked
        self.indexes = self.indexes.compressed()

        # Set max_data to only take less than all the indexes
        if self.max_data > len(self.indexes):
            self.max_data = len(self.indexes)
        if self.shuffle:
            if seed is not None:
                np.random.seed(seed)
            np.random.shuffle(self.indexes)
        # Build first sample
        self.sample = self.indexes[0 : self.max_data]
        if not self.same_samples:
            self.start_idx = self.max_data
        if self.balance == "global":
            # Compute effective labels that will be used for training
            if self.same_samples:
                y_eff = self.masked_labels[self.sample].compressed()
            else:
                y_eff = self.masked_labels[self.indexes].compressed()
            # Determine weights with effective labels
            bin_values, bin_edges = np.histogram(
                y_eff, bins=self.n_classes, range=(0, 1)
            )
            # Weight all labels for convenience
            bin_idx = np.digitize(self.labels, bin_edges)
            bin_idx[bin_idx == self.n_classes + 1] = self.n_classes
            bin_idx -= 1
            self.weights = len(y_eff) / (self.n_classes * bin_values[bin_idx])
        else:
            self.weights = np.ones(len(self.data))
        # Weight invalid labels to 0
        self.weights[self.masked_labels.mask] = 0

    def __len__(self):
        """Return length of generator.

        The length displayed is the length for the current epoch. Not the
        number of available windows accross multiple epochs.
        """
        return int(np.ceil(len(self.sample) / self.batch_size))

    def __getitem__(self, idx):
        """Get a batch of data.

        Parameters
        ----------
        idx : int
            Index of the batch to extract
        """
        # Get window center idxes
        batch_idxes = self.sample[idx * self.batch_size : (idx + 1) * self.batch_size]
        # Get full window idxes
        window_indices = batch_idxes.reshape(-1, 1) + np.arange(
            -(self.winsize // 2), (self.winsize - 1) // 2 + 1
        ).reshape(1, -1)
        batch_x = self.data[window_indices]
        # Determine head indices for labels
        if self.head_interval:
            head_indices = batch_idxes.reshape(-1, 1) + np.arange(
                -(self.winsize // 2), (self.winsize - 1) // 2 + 1, self.head_interval
            ).reshape(1, -1)
        else:
            head_indices = batch_idxes
        # Optionally reverse complement all or part of the sequences
        if self.strand == "rev":
            batch_x = batch_x[:, ::-1, ::-1]
            if self.head_interval:
                head_indices = head_indices[:, ::-1] + self.head_interval - 1
        elif self.strand == "both":
            half_size = self.batch_size // 2
            batch_x[:half_size] = batch_x[:half_size, ::-1, ::-1]
            if self.head_interval:
                head_indices[:half_size] = (
                    head_indices[:half_size, ::-1] + self.head_interval - 1
                )
        # Optionally add dimensions
        if self.extradims:
            batch_x = np.expand_dims(batch_x, axis=self.extradims)
        # Get y after optionnally reversing head_indices
        batch_y = self.labels[head_indices]
        # Make batch_y 2D (not sure if useful)
        if len(batch_y.shape) == 1:
            batch_y = np.expand_dims(batch_y, axis=1)
        # Divide continuous labels into classes and balance weights
        if self.balance == "batch":
            # Flatten in case of multiple outputs
            flat_batch_y = batch_y.ravel()
            # Compute batch weights based on valid labels
            batch_masked_y = self.masked_labels[head_indices]
            batch_y_eff = batch_masked_y.compressed()  # flattens
            bin_values, bin_edges = np.histogram(
                batch_y_eff, bins=self.n_classes, range=(0, 1)
            )
            bin_idx = np.digitize(flat_batch_y, bin_edges)
            bin_idx[bin_idx == self.n_classes + 1] = self.n_classes
            bin_idx -= 1
            batch_weights = len(batch_y_eff) / (self.n_classes * bin_values[bin_idx])
            # Weight invalid labels to 0
            batch_weights[batch_masked_y.mask.ravel()] = 0
            # Reshape as batch_y
            batch_weights = batch_weights.reshape(batch_y.shape)
        else:
            batch_weights = self.weights[head_indices]
        return batch_x, batch_y, batch_weights

    def on_epoch_end(self):
        """Update the sample to use for next epoch.

        The sample can be different from the one used in the previous epoch if
        there are enough windows. If all windows have been seen, a shuffle may
        be applied and additional windows are extracted from the start.
        """
        if self.same_samples:
            if self.shuffle:
                np.random.shuffle(self.sample)
        else:
            stop_idx = self.start_idx + self.max_data
            self.sample = self.indexes[self.start_idx : stop_idx]
            if stop_idx >= len(self.indexes):
                print("full data loop")
                # Complete sample by going back to the beginning of indexes
                if self.shuffle:
                    # Save current sample because shuffling will modify it
                    self.sample = self.sample.copy()
                    np.random.shuffle(self.indexes)
                stop_idx = stop_idx - len(self.indexes)
                if stop_idx != 0:
                    self.sample = np.concatenate((self.sample, self.indexes[:stop_idx]))
            # Update start_idx for next call to on_epoch_end
            self.start_idx = stop_idx


class PredGenerator(Sequence):
    def __init__(self, data, winsize, batch_size, extradims=None):
        self.data = data
        self.winsize = winsize
        self.batch_size = batch_size
        self.indexes = np.arange(self.winsize // 2, len(data) - (self.winsize // 2))
        self.extradims = extradims

    def __len__(self):
        return int(np.ceil(len(self.data) / self.batch_size))

    def __getitem__(self, idx):
        # Get window center idxes
        batch_idxes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        # Get full window idxes
        window_indices = batch_idxes.reshape(-1, 1) + np.arange(
            -(self.winsize // 2), self.winsize // 2 + 1
        ).reshape(1, -1)
        batch_x = self.data[window_indices]
        if self.extradims is not None:
            batch_x = np.expand_dims(batch_x, axis=self.extradims)
        batch_y = np.zeros((len(batch_x), 1))
        return batch_x, batch_y


def mae_cor(y_true, y_pred):
    """Compute loss with Mean absolute error and correlation.
    :Example:
    >>> model.compile(optimizer = 'adam', losses = mae_cor)
    >>> load_model('file', custom_objects = {'mae_cor : mae_cor})
    """
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X * Y)
    sigma_X = K.sqrt(K.sum(X * X))
    sigma_Y = K.sqrt(K.sum(Y * Y))

    cor = sigma_XY / (sigma_X * sigma_Y + K.epsilon())
    mae = K.mean(K.abs(y_true - y_pred))

    return (1 - cor) + mae


def correlate(y_true, y_pred):
    """Calculate the correlation between the predictions and the labels.
    :Example:
    >>> model.compile(optimizer = 'adam', losses = correlate)
    >>> load_model('file', custom_objects = {'correlate : correlate})
    """
    X = y_true - K.mean(y_true)
    Y = y_pred - K.mean(y_pred)

    sigma_XY = K.sum(X * Y)
    sigma_X = K.sqrt(K.sum(X * X))
    sigma_Y = K.sqrt(K.sum(Y * Y))

    return sigma_XY / (sigma_X * sigma_Y + K.epsilon())


def predict(
    model,
    one_hot_chr,
    winsize,
    head_interval=None,
    reverse=False,
    batch_size=1024,
    middle=False,
    extradims=None,
    order="ACGT",
):
    if winsize > len(one_hot_chr):
        raise ValueError("sequence too small")
    if reverse:
        if order == "ACGT":
            one_hot_chr = one_hot_chr[::-1, ::-1]
        else:
            one_hot_chr = utils.RC_one_hot(one_hot_chr, order)
    pred = np.zeros(len(one_hot_chr), dtype="float32")
    if head_interval is not None and middle:
        X = utils.strided_sliding_window_view(
            one_hot_chr, (winsize, 4), stride=winsize // 2, sliding_len=head_interval
        ).reshape(-1, winsize, 4)
        n_heads = winsize // head_interval
        y = model.predict(X).squeeze()[:, n_heads // 4 : 3 * n_heads // 4]
        y = np.transpose(y.reshape(-1, head_interval, n_heads // 2), [0, 2, 1]).ravel()
        pred[winsize // 4 : len(y) + winsize // 4] = y
        # Get last window
        leftover = len(pred) - (len(y) + winsize // 4)
        min_leftover = winsize // 4 + head_interval - 1
        if leftover > min_leftover:
            X = utils.strided_sliding_window_view(
                one_hot_chr[-winsize - head_interval + 1 :],
                (winsize, 4),
                stride=winsize // 2,
                sliding_len=head_interval,
            ).squeeze()
            y = model.predict(X).squeeze().T.ravel()
            pred[-leftover:-min_leftover] = y[-leftover + min_leftover :]
    elif head_interval is not None:
        X = utils.strided_sliding_window_view(
            one_hot_chr, (winsize, 4), stride=winsize, sliding_len=head_interval
        ).reshape(-1, winsize, 4)
        y = model.predict(X).squeeze()
        n_heads = y.shape[-1]
        y = np.transpose(y.reshape(-1, head_interval, n_heads), [0, 2, 1]).ravel()
        pred[: len(y)] = y
        # Get last_window
        leftover = len(pred) - len(y)
        if leftover > head_interval - 1:
            X = utils.strided_sliding_window_view(
                one_hot_chr[-winsize - head_interval + 1 :],
                (winsize, 4),
                stride=winsize,
                sliding_len=head_interval,
            ).squeeze()
            y = model.predict(X).squeeze().T.ravel()
            pred[-leftover : -head_interval + 1] = y[-leftover + head_interval - 1 :]
    else:
        X = PredGenerator(one_hot_chr, winsize, batch_size, extradims)
        pred[winsize // 2 : -(winsize // 2)] = model.predict(X).ravel()
    if reverse:
        return pred[::-1]
    else:
        return pred
