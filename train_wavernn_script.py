import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
import sys

import argparse


# utils.paths
import os


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""
    def __init__(self, data_path, voc_id, tts_id):
        self.base = Path(__file__).parent.parent.expanduser().resolve()

        # Data Paths
        self.data = Path(data_path).expanduser().resolve()
        self.quant = self.data/'quant'
        self.mel = self.data/'mel'
        self.gta = self.data/'gta'

        # WaveRNN/Vocoder Paths
        self.voc_checkpoints = self.base/'checkpoints'/f'{voc_id}.wavernn'
        self.voc_latest_weights = self.voc_checkpoints/'latest_weights.pyt'
        self.voc_latest_optim = self.voc_checkpoints/'latest_optim.pyt'
        self.voc_output = self.base/'model_outputs'/f'{voc_id}.wavernn'
        self.voc_step = self.voc_checkpoints/'step.npy'
        self.voc_log = self.voc_checkpoints/'log.txt'

        # Tactron/TTS Paths
        self.tts_checkpoints = self.base/'checkpoints'/f'{tts_id}.tacotron'
        self.tts_latest_weights = self.tts_checkpoints/'latest_weights.pyt'
        self.tts_latest_optim = self.tts_checkpoints/'latest_optim.pyt'
        self.tts_output = self.base/'model_outputs'/f'{tts_id}.tacotron'
        self.tts_step = self.tts_checkpoints/'step.npy'
        self.tts_log = self.tts_checkpoints/'log.txt'
        self.tts_attention = self.tts_checkpoints/'attention'
        self.tts_mel_plot = self.tts_checkpoints/'mel_plots'

        self.create_paths()

    def create_paths(self):
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)
        os.makedirs(self.tts_checkpoints, exist_ok=True)
        os.makedirs(self.tts_output, exist_ok=True)
        os.makedirs(self.tts_attention, exist_ok=True)
        os.makedirs(self.tts_mel_plot, exist_ok=True)

    def get_tts_named_weights(self, name):
        """Gets the path for the weights in a named tts checkpoint."""
        return self.tts_checkpoints/f'{name}_weights.pyt'

    def get_tts_named_optim(self, name):
        """Gets the path for the optimizer state in a named tts checkpoint."""
        return self.tts_checkpoints/f'{name}_optim.pyt'

    def get_voc_named_weights(self, name):
        """Gets the path for the weights in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_weights.pyt'

    def get_voc_named_optim(self, name):
        """Gets the path for the optimizer state in a named voc checkpoint."""
        return self.voc_checkpoints/f'{name}_optim.pyt'


# utils.checkpoints
def get_checkpoint_paths(checkpoint_type: str, paths: Paths):
    """
    Returns the correct checkpointing paths
    depending on whether model is Vocoder or TTS

    Args:
        checkpoint_type: Either 'voc' or 'tts'
        paths: Paths object
    """
    if checkpoint_type is 'tts':
        weights_path = paths.tts_latest_weights
        optim_path = paths.tts_latest_optim
        checkpoint_path = paths.tts_checkpoints
    elif checkpoint_type is 'voc':
        weights_path = paths.voc_latest_weights
        optim_path = paths.voc_latest_optim
        checkpoint_path = paths.voc_checkpoints
    else:
        raise NotImplementedError

    return weights_path, optim_path, checkpoint_path


def save_checkpoint(checkpoint_type: str, paths: Paths, model, optimizer, *,
        name=None, is_silent=False):
    """Saves the training session to disk.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` or `WaveRNN` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        name:  If provided, will name to a checkpoint with the given name. Note
            that regardless of whether this is provided or not, this function
            will always update the files specified in `paths` that give the
            location of the latest weights and optimizer state. Saving
            a named checkpoint happens in addition to this update.
    """
    def helper(path_dict, is_named):
        s = 'named' if is_named else 'latest'
        num_exist = sum(p.exists() for p in path_dict.values())

        if num_exist not in (0,2):
            # Checkpoint broken
            raise FileNotFoundError(
                f'We expected either both or no files in the {s} checkpoint to '
                'exist, but instead we got exactly one!')

        if num_exist == 0:
            if not is_silent: print(f'Creating {s} checkpoint...')
            for p in path_dict.values():
                p.parent.mkdir(parents=True, exist_ok=True)
        else:
            if not is_silent: print(f'Saving to existing {s} checkpoint...')

        if not is_silent: print(f'Saving {s} weights: {path_dict["w"]}')
        model.save(path_dict['w'])
        if not is_silent: print(f'Saving {s} optimizer state: {path_dict["o"]}')
        torch.save(optimizer.state_dict(), path_dict['o'])

    weights_path, optim_path, checkpoint_path = \
        get_checkpoint_paths(checkpoint_type, paths)

    latest_paths = {'w': weights_path, 'o': optim_path}
    helper(latest_paths, False)

    if name:
        named_paths = {
            'w': checkpoint_path/f'{name}_weights.pyt',
            'o': checkpoint_path/f'{name}_optim.pyt',
        }
        helper(named_paths, True)


def restore_checkpoint(checkpoint_type: str, paths: Paths, model, optimizer, *,
        name=None, create_if_missing=False):
    """Restores from a training session saved to disk.

    NOTE: The optimizer's state is placed on the same device as it's model
    parameters. Therefore, be sure you have done `model.to(device)` before
    calling this method.

    Args:
        paths:  Provides information about the different paths to use.
        model:  A `Tacotron` or `WaveRNN` model to save the parameters and buffers from.
        optimizer:  An optmizer to save the state of (momentum, etc).
        name:  If provided, will restore from a checkpoint with the given name.
            Otherwise, will restore from the latest weights and optimizer state
            as specified in `paths`.
        create_if_missing:  If `True`, will create the checkpoint if it doesn't
            yet exist, as well as update the files specified in `paths` that
            give the location of the current latest weights and optimizer state.
            If `False` and the checkpoint doesn't exist, will raise a
            `FileNotFoundError`.
    """

    weights_path, optim_path, checkpoint_path = \
        get_checkpoint_paths(checkpoint_type, paths)

    if name:
        path_dict = {
            'w': checkpoint_path/f'{name}_weights.pyt',
            'o': checkpoint_path/f'{name}_optim.pyt',
        }
        s = 'named'
    else:
        path_dict = {
            'w': weights_path,
            'o': optim_path
        }
        s = 'latest'

    num_exist = sum(p.exists() for p in path_dict.values())
    if num_exist == 2:
        # Checkpoint exists
        print(f'Restoring from {s} checkpoint...')
        print(f'Loading {s} weights: {path_dict["w"]}')
        model.load(path_dict['w'])
        print(f'Loading {s} optimizer state: {path_dict["o"]}')
        optimizer.load_state_dict(torch.load(path_dict['o']))
    elif create_if_missing:
        save_checkpoint(checkpoint_type, paths, model, optimizer, name=name, is_silent=False)
    else:
        raise FileNotFoundError(f'The {s} checkpoint could not be found!')


# utils
def data_parallel_workaround(model, *input):
    global _output_ref
    global _replicas_ref
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    # input.shape = (num_args, batch, ...)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    # inputs.shape = (num_gpus, num_args, batch/num_gpus, ...)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    _output_ref = outputs
    _replicas_ref = replicas
    return y_hat


# utils.display
def stream(message):
    sys.stdout.write(f"\r{message}")


def simple_table(item_tuples):

    border_pattern = '+---------------------------------------'
    whitespace = '                                            '

    headings, cells, = [], []

    for item in item_tuples:

        heading, cell = str(item[0]), str(item[1])

        pad_head = True if len(heading) < len(cell) else False

        pad = abs(len(heading) - len(cell))
        pad = whitespace[:pad]

        pad_left = pad[:len(pad)//2]
        pad_right = pad[len(pad)//2:]

        if pad_head:
            heading = pad_left + heading + pad_right
        else:
            cell = pad_left + cell + pad_right

        headings += [heading]
        cells += [cell]

    border, head, body = '', '', ''

    for i in range(len(item_tuples)):

        temp_head = f'| {headings[i]} '
        temp_body = f'| {cells[i]} '

        border += border_pattern[:len(temp_head)]
        head += temp_head
        body += temp_body

        if i == len(item_tuples) - 1:
            head += '|'
            body += '|'
            border += '+'

    print(border)
    print(head)
    print(border)
    print(body)
    print(border)
    print(' ')


# utils.dataset
import pickle
import random
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


class VocoderDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, train_gta=False):
        self.metadata = dataset_ids
        self.mel_path = path/'gta' if train_gta else path/'mel'
        self.quant_path = path/'quant'

    def __getitem__(self, index):
        item_id = self.metadata[index]
        m = np.load(self.mel_path/f'{item_id}.npy')
        x = np.load(self.quant_path/f'{item_id}.npy')
        return m, x

    def __len__(self):
        return len(self.metadata)


def label_2_float(x, bits):
    return 2 * x / (2**bits - 1.) - 1.


def collate_vocoder(batch):
    mel_win = hp.voc_seq_len // hp.hop_length + 2 * hp.voc_pad
    max_offsets = [x[0].shape[-1] -2 - (mel_win + 2 * hp.voc_pad) for x in batch]
    mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
    sig_offsets = [(offset + hp.voc_pad) * hp.hop_length for offset in mel_offsets]

    mels = [x[0][:, mel_offsets[i]:mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

    labels = [x[1][sig_offsets[i]:sig_offsets[i] + hp.voc_seq_len + 1] for i, x in enumerate(batch)]

    mels = np.stack(mels).astype(np.float32)
    labels = np.stack(labels).astype(np.int64)

    mels = torch.tensor(mels)
    labels = torch.tensor(labels).long()

    x = labels[:, :hp.voc_seq_len]
    y = labels[:, 1:]

    bits = 16 if hp.voc_mode == 'MOL' else hp.bits

    x = label_2_float(x.float(), bits)

    if hp.voc_mode == 'MOL':
        y = label_2_float(y.float(), bits)

    return x, y, mels


def get_vocoder_datasets(path: Path, batch_size, train_gta):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]

    random.seed(1234)
    random.shuffle(dataset_ids)

    test_ids = dataset_ids[-hp.voc_test_samples:]
    train_ids = dataset_ids[:-hp.voc_test_samples]

    train_dataset = VocoderDataset(path, train_ids, train_gta)
    test_dataset = VocoderDataset(path, test_ids, train_gta)

    train_set = DataLoader(train_dataset,
                           collate_fn=collate_vocoder,
                           batch_size=batch_size,
                           num_workers=2,
                           shuffle=True,
                           pin_memory=True)

    test_set = DataLoader(test_dataset,
                          batch_size=1,
                          num_workers=1,
                          shuffle=False,
                          pin_memory=True)

    return train_set, test_set


# utils.distribution
def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def discretized_mix_logistic_loss(y_hat, y, num_classes=65536,
                                  log_scale_min=None, reduce=True):
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    y_hat = y_hat.permute(0,2,1)
    assert y_hat.dim() == 3
    assert y_hat.size(1) % 3 == 0
    nr_mix = y_hat.size(1) // 3

    # (B x T x C)
    y_hat = y_hat.transpose(1, 2)

    # unpack parameters. (B, T, num_mixtures) x 3
    logit_probs = y_hat[:, :, :nr_mix]
    means = y_hat[:, :, nr_mix:2 * nr_mix]
    log_scales = torch.clamp(y_hat[:, :, 2 * nr_mix:3 * nr_mix], min=log_scale_min)

    # B x T x 1 -> B x T x num_mixtures
    y = y.expand_as(means)

    centered_y = y - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = torch.sigmoid(min_in)

    # log probability for edge case of 0 (before scaling)
    # equivalent: torch.log(F.sigmoid(plus_in))
    log_cdf_plus = plus_in - F.softplus(plus_in)

    # log probability for edge case of 255 (before scaling)
    # equivalent: (1 - F.sigmoid(min_in)).log()
    log_one_minus_cdf_min = -F.softplus(min_in)

    # probability for all other cases
    cdf_delta = cdf_plus - cdf_min

    mid_in = inv_stdv * centered_y
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # tf equivalent
    """
    log_probs = tf.where(x < -0.999, log_cdf_plus,
                         tf.where(x > 0.999, log_one_minus_cdf_min,
                                  tf.where(cdf_delta > 1e-5,
                                           tf.log(tf.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log(127.5))))
    """
    # TODO: cdf_delta <= 1e-5 actually can happen. How can we choose the value
    # for num_classes=65536 case? 1e-7? not sure..
    inner_inner_cond = (cdf_delta > 1e-5).float()

    inner_inner_out = inner_inner_cond * \
        torch.log(torch.clamp(cdf_delta, min=1e-12)) + \
        (1. - inner_inner_cond) * (log_pdf_mid - np.log((num_classes - 1) / 2))
    inner_cond = (y > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (y < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out

    log_probs = log_probs + F.log_softmax(logit_probs, -1)

    if reduce:
        return -torch.mean(log_sum_exp(log_probs))
    else:
        return -log_sum_exp(log_probs).unsqueeze(-1)


# utils -> hparams
import re
from typing import Union
from importlib.util import spec_from_file_location, module_from_spec


def _import_from_file(name, path: Path):
    """Programmatically returns a module object from a filepath"""
    if not Path(path).exists():
        raise FileNotFoundError('"%s" doesn\'t exist!' % path)
    spec = spec_from_file_location(name, path)
    if spec is None:
        raise ValueError('could not load module from "%s"' % path)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class __HParams:
    """Manages the hyperparams pseudo-module"""
    def __init__(self, path: Union[str, Path]=None):
        """Constructs the hyperparameters from a path to a python module. If
        `path` is None, will raise an AttributeError whenever its attributes
        are accessed. Otherwise, configures self based on `path`."""
        if path is None:
            self._configured = False
        else:
            self.configure(path)

    def __getattr__(self, item):
        if not self.is_configured():
            raise AttributeError("HParams not configured yet. Call self.configure()")
        else:
            return super().__getattr__(item)

    def configure(self, path: Union[str, Path]):
        """Configures hparams by copying over atrributes from a module with the
        given path. Raises an exception if already configured."""
        if self.is_configured():
            raise RuntimeError("Cannot reconfigure hparams!")

        ###### Check for proper path ######
        if not isinstance(path, Path):
            path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Could not find hparams file {path}")
        elif path.suffix != ".py":
            raise ValueError("`path` must be a python file")

        ###### Load in attributes from module ######
        m = _import_from_file("hparams", path)

        reg = re.compile(r"^__.+__$")  # Matches magic methods
        for name, value in m.__dict__.items():
            if reg.match(name):
                # Skip builtins
                continue
            if name in self.__dict__:
                # Cannot overwrite already existing attributes
                raise AttributeError(
                    f"module at `path` cannot contain attribute {name} as it "
                    "overwrites an attribute of the same name in utils.hparams")
            # Fair game to copy over the attribute
            self.__setattr__(name, value)

        self._configured = True

    def is_configured(self):
        return self._configured


hp = __HParams()


# models.fatchord_version
import librosa
import math
import torch.nn as nn


def progbar(i, n, size=16):
    done = (i * size) // n
    bar = ''
    for i in range(size):
        bar += '█' if i <= done else '░'
    return bar


def save_wav(x, path):
    librosa.output.write_wav(path, x.astype(np.float32), sr=hp.sample_rate)


def sample_from_discretized_mix_logistic(y, log_scale_min=None):
    """
    Sample from discretized mixture of logistic distributions
    Args:
        y (Tensor): B x C x T
        log_scale_min (float): Log scale minimum value
    Returns:
        Tensor: sample in range of [-1, 1].
    """
    if log_scale_min is None:
        log_scale_min = float(np.log(1e-14))
    assert y.size(1) % 3 == 0
    nr_mix = y.size(1) // 3

    # B x T x C
    y = y.transpose(1, 2)
    logit_probs = y[:, :, :nr_mix]

    # sample mixture indicator from softmax
    temp = logit_probs.data.new(logit_probs.size()).uniform_(1e-5, 1.0 - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=-1)

    # (B, T) -> (B, T, nr_mix)
    one_hot = F.one_hot(argmax, nr_mix).float()
    # select logistic parameters
    means = torch.sum(y[:, :, nr_mix:2 * nr_mix] * one_hot, dim=-1)
    log_scales = torch.clamp(torch.sum(
        y[:, :, 2 * nr_mix:3 * nr_mix] * one_hot, dim=-1), min=log_scale_min)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = means.data.new(means.size()).uniform_(1e-5, 1.0 - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))

    x = torch.clamp(torch.clamp(x, min=-1.), max=1.)

    return x


def decode_mu_law(y, mu, from_labels=True):
    # TODO: get rid of log2 - makes no sense
    if from_labels: y = label_2_float(y, math.log2(mu))
    mu = mu - 1
    x = np.sign(y) / mu * ((1 + mu) ** np.abs(y) - 1)
    return x


class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for i in range(res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers: x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(self, feat_dims, upsample_scales, compute_dims,
                 res_blocks, res_out_dims, pad):
        super().__init__()
        total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * total_scale
        self.resnet = MelResNet(res_blocks, feat_dims, compute_dims, res_out_dims, pad)
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


class WaveRNN(nn.Module):
    def __init__(self, rnn_dims, fc_dims, bits, pad, upsample_factors,
                 feat_dims, compute_dims, res_out_dims, res_blocks,
                 hop_length, sample_rate, mode='RAW'):
        super().__init__()
        self.mode = mode
        self.pad = pad
        if self.mode == 'RAW':
            self.n_classes = 2 ** bits
        elif self.mode == 'MOL':
            self.n_classes = 30
        else:
            RuntimeError("Unknown model mode value - ", self.mode)

        # List of rnns to call `flatten_parameters()` on
        self._to_flatten = []

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.upsample = UpsampleNetwork(feat_dims, upsample_factors, compute_dims, res_blocks, res_out_dims, pad)
        self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)

        self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
        self.rnn2 = nn.GRU(rnn_dims + self.aux_dims, rnn_dims, batch_first=True)
        self._to_flatten += [self.rnn1, self.rnn2]

        self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
        self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
        self.fc3 = nn.Linear(fc_dims, self.n_classes)

        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.num_params()

        # Avoid fragmentation of RNN parameters and associated warning
        self._flatten_parameters()

    def forward(self, x, mels):
        device = next(self.parameters()).device  # use same device as parameters

        # Although we `_flatten_parameters()` on init, when using DataParallel
        # the model gets replicated, making it no longer guaranteed that the
        # weights are contiguous in GPU memory. Hence, we must call it again
        self._flatten_parameters()

        self.step += 1
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        h2 = torch.zeros(1, bsize, self.rnn_dims, device=device)
        mels, aux = self.upsample(mels)

        aux_idx = [self.aux_dims * i for i in range(5)]
        a1 = aux[:, :, aux_idx[0]:aux_idx[1]]
        a2 = aux[:, :, aux_idx[1]:aux_idx[2]]
        a3 = aux[:, :, aux_idx[2]:aux_idx[3]]
        a4 = aux[:, :, aux_idx[3]:aux_idx[4]]

        x = torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
        x = self.I(x)
        res = x
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2)
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2)
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def generate(self, mels, save_path: Union[str, Path], batched, target, overlap, mu_law):
        self.eval()

        device = next(self.parameters()).device  # use same device as parameters

        mu_law = mu_law if self.mode == 'RAW' else False

        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():

            mels = torch.as_tensor(mels, device=device)
            wave_len = (mels.size(-1) - 1) * self.hop_length
            mels = self.pad_tensor(mels.transpose(1, 2), pad=self.pad, side='both')
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims, device=device)
            h2 = torch.zeros(b_size, self.rnn_dims, device=device)
            x = torch.zeros(b_size, 1, device=device)

            d = self.aux_dims
            aux_split = [aux[:, :, d * i:d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                a1_t, a2_t, a3_t, a4_t = \
                    (a[:, i, :] for a in aux_split)

                x = torch.cat([x, m_t, a1_t], dim=1)
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1)
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1)
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1)
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == 'MOL':
                    sample = sample_from_discretized_mix_logistic(logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    # x = torch.FloatTensor([[sample]]).cuda()
                    x = sample.transpose(0, 1)

                elif self.mode == 'RAW':
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.) - 1.
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError("Unknown model mode value - ", self.mode)

                if i % 100 == 0: self.gen_display(i, seq_len, b_size, start)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu().numpy()
        output = output.astype(np.float64)

        if mu_law:
            output = decode_mu_law(output, self.n_classes, False)

        if batched:
            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]
        output[-20 * self.hop_length:] *= fade_out

        save_wav(output, save_path)

        self.train()

        return output


    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        pbar = progbar(i, seq_len)
        msg = f'| {pbar} {i*b_size}/{seq_len*b_size} | Batch Size: {b_size} | Gen Rate: {gen_rate:.1f}kHz | '
        stream(msg)

    def get_gru_cell(self, gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    def pad_tensor(self, x, pad, side='both'):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == 'both' else t + pad
        padded = torch.zeros(b, total, c, device=x.device)
        if side == 'before' or side == 'both':
            padded[:, pad:pad + t, :] = x
        elif side == 'after':
            padded[:, :t, :] = x
        return padded

    def fold_with_overlap(self, x, target, overlap):

        ''' Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()

        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)

        Details:
            x = [[h1, h2, ... hn]]

            Where each h is a vector of conditioning features

            Eg: target=2, overlap=1 with x.size(1)=10

            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        '''

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side='after')

        folded = torch.zeros(num_folds, target + 2 * overlap, features, device=x.device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    def xfade_and_unfold(self, y, target, overlap):

        ''' Applies a crossfade and unfolds into a 1d array.

        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup

        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64

        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]

            Apply a gain envelope at both ends of the sequences

            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]

            Stagger and add up the groups of samples:

            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]

        '''

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)
        linear = np.ones((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([linear, fade_out])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def get_step(self):
        return self.step.data.item()

    def log(self, path, msg):
        with open(path, 'a') as f:
            print(msg, file=f)

    def load(self, path: Union[str, Path]):
        # Use device of model params as location for loaded state
        device = next(self.parameters()).device
        self.load_state_dict(torch.load(path, map_location=device), strict=False)

    def save(self, path: Union[str, Path]):
        # No optimizer argument because saving a model should not include data
        # only relevant in the training process - it should only be properties
        # of the model itself. Let caller take care of saving optimzier state.
        torch.save(self.state_dict(), path)

    def num_params(self, print_out=True):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        if print_out:
            print('Trainable Parameters: %.3fM' % parameters)
        return parameters

    def _flatten_parameters(self):
        """Calls `flatten_parameters` on all the rnns used by the WaveRNN. Used
        to improve efficiency and avoid PyTorch yelling at us."""
        [m.flatten_parameters() for m in self._to_flatten]


# gen_wavernn
def gen_testset(model: WaveRNN, test_set, samples, batched, target, overlap, save_path: Path):

    k = model.get_step() // 1000

    for i, (m, x) in enumerate(test_set, 1):

        if i > samples: break

        print('\n| Generating: %i/%i' % (i, samples))

        x = x[0].numpy()

        bits = 16 if hp.voc_mode == 'MOL' else hp.bits

        if hp.mu_law and hp.voc_mode != 'MOL':
            x = decode_mu_law(x, 2**bits, from_labels=True)
        else:
            x = label_2_float(x, bits)

        save_wav(x, save_path/f'{k}k_steps_{i}_target.wav')

        batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
        save_str = str(save_path/f'{k}k_steps_{i}_{batch_str}.wav')

        _ = model.generate(m, save_str, batched, target, overlap, hp.mu_law)


def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # load hparams from file
    if args.lr is None:
        args.lr = hp.voc_lr
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count() != 0:
            raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    optimizer = optim.Adam(voc_model.parameters())
    restore_checkpoint('voc', paths, voc_model, optimizer, create_if_missing=True)

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, lr, total_steps)

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')


def voc_train_loop(paths: Paths, model: WaveRNN, loss_func, optimizer, train_set, test_set, lr, total_steps):
    # Use same device as model parameters
    device = next(model.parameters()).device

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    for e in range(1, epochs + 1):

        start = time.time()
        running_loss = 0.

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                y_hat = data_parallel_workaround(model, x, m)
            else:
                y_hat = model(x, m)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)


            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                if np.isnan(grad_norm.cpu()):
                    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint('voc', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('voc', paths, model, optimizer, is_silent=True)
        model.log(paths.voc_log, msg)
        print(' ')


if __name__ == "__main__":
    main()
