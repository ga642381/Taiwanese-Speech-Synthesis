import argparse
from csv import DictReader
import glob
from multiprocessing import Pool, cpu_count
from os.path import splitext, basename
from pathlib import Path
import pickle
import numpy as np
from typing import Union
from utils import hparams as hp
from utils.display import *
from utils.dsp import *
from utils.files import get_files
from utils.paths import Paths
from utils.display import simple_table
from functools import partial

from 臺灣言語工具.解析整理.拆文分析器 import 拆文分析器
from 臺灣言語工具.語音合成 import 台灣話口語講法


def valid_n_workers(num):
    n = int(num)
    if n < 1:
        raise argparse.ArgumentTypeError(
            '%r must be an integer greater than 0' % num)
    return n


def wav_to_mel_quant(path: Path):
    y = load_wav(path)
    # === 1. wav normalization === #
    peak = np.abs(y).max()
    if hp.peak_norm or peak > 1.0:
        y /= peak

    # === 2. mel === #
    mel = melspectrogram(y)

    # === 3. quant === #
    if hp.voc_mode == 'RAW':
        if hp.mu_law:
            quant = encode_mu_law(y, mu=2**hp.bits)
        else:
            quant = float_2_label(y, bits=hp.bits)

    elif hp.voc_mode == 'MOL':
        quant = float_2_label(y, bits=16)

    return mel.astype(np.float32), quant.astype(np.int64)


class Preprocessor():
    def __init__(self, args):
        self.args = args
        self.config()

        # === args === #
        self.audio_extension = args.extension
        self.dataset_path = args.data_dir
        self.n_workers = max(1, args.num_workers)
        self.paths = Paths(args.save_dir, hp.voc_model_id, hp.tts_model_id)

    def config(self):
        hp.configure(self.args.hp_file)

    # === Dataset === #
    def suisiann(self, path: Union[str, Path], wav_files):
        csv_file = get_files(path, extension='.csv')

        assert len(csv_file) == 1

        u_tihleh = set()
        for sootsai in wav_files:
            u_tihleh.add(basename(sootsai))
        text_dict = {}

        with open(csv_file[0], encoding='utf-8') as f:
            for tsua in DictReader(f):
                mia = basename(tsua['音檔'])
                if mia in u_tihleh:
                    imtong = splitext(mia)[0]
                    hj = tsua['漢字']
                    lmj = tsua['羅馬字']
                    text = 台灣話口語講法(
                        拆文分析器.建立句物件(hj, lmj)
                    )
                    text_dict[imtong] = text

        return text_dict

    def process_wav(self, path: Path):
        wav_id = path.stem
        m, x = wav_to_mel_quant(path)
        np.save(self.paths.mel / f'{wav_id}.npy', m, allow_pickle=False)
        np.save(self.paths.quant / f'{wav_id}.npy', x, allow_pickle=False)
        return wav_id, m.shape[-1]

    def run(self):
        # === wav files === #
        wav_files = get_files(
            self.dataset_path, extension=self.audio_extension)
        print(
            f'\n{len(wav_files)} {self.audio_extension[1:]} files found in "{self.dataset_path}"\n')

        if len(wav_files) == 0:
            print(
                f'no wav file found in {self.dataset_path}, please check specify --data_dir correctly!.\n')

        # === process === #
        else:
            if not hp.ignore_tts:
                text_dict = self.suisiann(self.dataset_path, wav_files)
                with open(self.paths.data / 'text_dict.pkl', 'wb') as f:
                    pickle.dump(text_dict, f)

            simple_table([
                ('Sample Rate', hp.sample_rate),
                ('Bit Depth',   hp.bits),
                ('Mu Law',      hp.mu_law),
                ('Hop Length',  hp.hop_length),
                ('CPU Usage', f'{self.n_workers}/{cpu_count()}')
            ])

            pool = Pool(processes=self.n_workers)

            dataset = []

            #func = partial(process_wav, self.paths)
            for i, (item_id, length) in enumerate(pool.imap_unordered(self.process_wav, wav_files), 1):
                dataset += [(item_id, length)]
                bar = progbar(i, len(wav_files))
                message = f'{bar} {i}/{len(wav_files)} '
                stream(message)

            with open(self.paths.data / 'dataset.pkl', 'wb') as f:
                pickle.dump(dataset, f)

            print(
                '\n\nCompleted. Ready to run "python train_tacotron2.py" or "python train_wavernn.py". \n')


def main(args):
    P = Preprocessor(args)
    P.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocessing for WaveRNN and Tacotron2')

    parser.add_argument("data_dir", type=str)
    parser.add_argument("save_dir", type=str)

    parser.add_argument(
        '--extension', '-e', metavar='EXT', default='.wav',
        help='file extension to search for in dataset folder'
    )

    parser.add_argument(
        '--num_workers', '-w', metavar='N', type=valid_n_workers,
        default=cpu_count() - 1,
        help='The number of worker threads to use for preprocessing'
    )

    parser.add_argument(
        '--hp_file', metavar='FILE', default='hparams.py',
        help='The file to use for the hyperparameters'
    )
    args = parser.parse_args()
    main(args)
