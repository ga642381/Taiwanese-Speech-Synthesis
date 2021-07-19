import pickle
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from utils.dsp import *
from utils import hparams as hp
from utils.text import text_to_sequence
from utils.paths import Paths
from pathlib import Path
import hparams
import os
###########################
# WaveRNN/Vocoder Dataset #
###########################
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

def get_possible(path, dataset_ids):
    mel_path = path/'gta'
    possible_files = os.listdir(mel_path)
    possible_ids = [os.path.splitext(f)[0] for f in possible_files]
    for _id in dataset_ids:
        if _id not in possible_ids: dataset_ids.remove(_id)
    return dataset_ids
def get_vocoder_datasets(path: Path, batch_size, train_gta):

    #with open(path/'dataset_wavernn.pkl', 'rb') as f:
    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = [x[0] for x in dataset]
    if train_gta:
        dataset_ids = get_possible(path, dataset_ids)
        print(f"{len(dataset_ids)} files found in {path}/gta ")
    assert(len(dataset_ids)!=0)

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


####################
# Tacotron Dataset #
####################

def get_tts_datasets(path: Path, batch_size, r):

    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    dataset_ids = []
    mel_lengths = []

    for (item_id, len) in dataset:
        if len <= hp.tts_max_mel_len:
            dataset_ids += [item_id] ###
            mel_lengths += [len]

    with open(path/'text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f) ###

    train_dataset = TTSDataset(path, dataset_ids, text_dict)

    sampler = None

    if hp.tts_bin_lengths:
        sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)

    train_set = DataLoader(train_dataset,
                           collate_fn=lambda batch: collate_tts(batch, r),
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=1,
                           pin_memory=True)

    longest = mel_lengths.index(max(mel_lengths))

    # Used to evaluate attention during training process
    attn_example = dataset_ids[longest]

    # print(attn_example)

    return train_set, attn_example


class TTSDataset(Dataset):
    def __init__(self, path: Path, dataset_ids, text_dict):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

    def __getitem__(self, index):
        item_id = self.metadata[index]
        x = text_to_sequence(self.text_dict[item_id], hp.tts_cleaner_names)
        mel = np.load(self.path/'mel'/f'{item_id}.npy')
        mel_len = mel.shape[-1]
        return x, mel, item_id, mel_len

    def __len__(self):
        return len(self.metadata)


def collate_tts(batch, r):

    x_lens = [len(x[0]) for x in batch]
    max_x_len = max(x_lens)

    chars = [pad1d(x[0], max_x_len) for x in batch]
    chars = np.stack(chars)

    spec_lens = [x[1].shape[-1] for x in batch]
    max_spec_len = max(spec_lens) + 1
    if max_spec_len % r != 0:
        max_spec_len += r - max_spec_len % r

    mel = [pad2d(x[1], max_spec_len) for x in batch]
    mel = np.stack(mel)

    ids = [x[2] for x in batch]
    mel_lens = [x[3] for x in batch]

    chars = torch.tensor(chars).long()
    mel = torch.tensor(mel)

    # scale spectrograms to -4 <--> 4
    mel = (mel * 8.) - 4.
    return chars, mel, ids, mel_lens

def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def pad2d(x, max_len):
    return np.pad(x, ((0, 0), (0, max_len - x.shape[-1])), mode='constant')

#####################
# Tacotron2 Dataset #
#####################

# https://github.com/NVIDIA/tacotron2
def get_tacotron2_datasets(path: Path, batch_size):
    with open(path/'dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)
    dataset_ids = []
    mel_lengths = []

    for (item_id, mel_len) in dataset:
        if mel_len <= hp.tts_max_mel_len:
            dataset_ids += [item_id]
            mel_lengths += [mel_len]

    with open(path/'text_dict.pkl', 'rb') as f:
        text_dict = pickle.load(f)

    #!
    train_dataset = TextMelLoader(path, dataset_ids, text_dict, hparams)
    collate_fn = TextMelCollate(hparams.n_frames_per_step)

    if hp.tts_bin_lengths:
        sampler = BinnedLengthSampler(mel_lengths, batch_size, batch_size * 3)
    else:
        sampler = None
    
    train_loader = DataLoader(train_dataset,
                              collate_fn=collate_fn,
                              batch_size=batch_size,
                              sampler=sampler,
                              num_workers=1,
                              pin_memory=True)

    longest = mel_lengths.index(max(mel_lengths))

    # Used to evaluate attention during training process
    attn_example = dataset_ids[longest]

    return train_loader, attn_example

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """
    def __init__(self, path: Path, dataset_ids, text_dict, hparams):
        self.path = path
        self.metadata = dataset_ids
        self.text_dict = text_dict

        self.text_cleaners = hparams.text_cleaners
        self.sampling_rate = hparams.sample_rate
        random.seed(1234)

    def get_mel_text_pair(self, index):
        # separate filename and text
        item_id = self.metadata[index]

        # === #
        text = self.get_text(item_id)
        mel  = self.get_mel(item_id)
        return (text, mel, item_id)

    def get_mel(self, item_id):
        # no load from disk
        filename = self.path/'mel'/f'{item_id}.npy'
        melspec = np.load(filename)
        melspec = torch.from_numpy(melspec)
        return melspec

    def get_text(self, item_id):
        x = text_to_sequence(self.text_dict[item_id], self.text_cleaners)
        x = torch.IntTensor(x)
        return x

    def __getitem__(self, index):
        return self.get_mel_text_pair(index)

    def __len__(self):
        return len(self.metadata)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized, item_id]
        """
        # === TEXT === #
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text
        
        # === MEL === #
        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)
        
        # === ITEM_ID === #
        item_ids = []
        for i in range(len(ids_sorted_decreasing)):
            item_ids.append(batch[ids_sorted_decreasing[i]][2])
            
            
        return item_ids, \
               text_padded, input_lengths, mel_padded, gate_padded, \
               output_lengths

        #  text_padded, input_lengths, mel_padded, gate_padded, output_lengths = batch

        
class BinnedLengthSampler(Sampler):
    def __init__(self, lengths, batch_size, bin_size):
        _, self.idx = torch.sort(torch.tensor(lengths).long())
        self.batch_size = batch_size
        self.bin_size = bin_size
        assert self.bin_size % self.batch_size == 0

    def __iter__(self):
        # Need to change to numpy since there's a bug in random.shuffle(tensor)
        # TODO: Post an issue on pytorch repo
        idx = self.idx.numpy()
        bins = []

        for i in range(len(idx) // self.bin_size):
            this_bin = idx[i * self.bin_size:(i + 1) * self.bin_size]
            random.shuffle(this_bin)
            bins += [this_bin]

        random.shuffle(bins)
        binned_idx = np.stack(bins).reshape(-1)

        if len(binned_idx) < len(idx):
            last_bin = idx[len(binned_idx):]
            random.shuffle(last_bin)
            binned_idx = np.concatenate([binned_idx, last_bin])

        return iter(torch.tensor(binned_idx).long())

    def __len__(self):
        return len(self.idx)
