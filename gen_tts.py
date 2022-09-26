import numpy as np
import torch
import os
import argparse
import re
from tacotron2 import Tacotron2
from wavernn import WaveRNN

from utils.text.symbols import symbols
from utils.paths import Paths

from utils.text import text_to_sequence
from utils.display import save_attention, simple_table
from utils.dsp import reconstruct_waveform, save_wav


from utils import hparams as hp
#import hparams as hp


class TaiwaneseTacotron():
    def __init__(self, args):
        self.args = args
        #================ vocoder ================#
        '''
            Setting up vocoder hyperparameters
            
            Supported vocoder:
                1. wavernn
                2. griffinlim
        '''
        if not (self.args.vocoder == "wavernn" or self.args.vocoder == "griffinlim"):
            raise argparse.ArgumentError('Must provide a valid vocoder type!')

        hp.configure(self.args.hp_file)  # Load hparams from file

        # set defaults for any arguments that depend on hparams
        if self.args.vocoder == 'wavernn':
            if self.args.target is None:
                self.args.target = hp.voc_target
            if self.args.overlap is None:
                self.args.overlap = hp.voc_overlap
            if self.args.batched is None:
                self.args.batched = hp.voc_gen_batched

        #================ others ================#
        # self.paths = Paths("", hp.voc_model_id, hp.tts_model_id, output_stage=True)

        # setup computing resource
        if not self.args.force_cpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        print('Using device:', device)

        # === Initialize Wavernn === #
        if self.args.vocoder == 'wavernn':
            print('\nInitializing WaveRNN Model...\n')
            self.voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
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

            # voc_load_path = self.args.voc_weights if self.args.voc_weights else self.paths.voc_latest_weights
            voc_load_path = self.args.voc_weights
            self.voc_model.load(voc_load_path)

        # === Initialize Tacotron2 === #
        print('\nInitializing Tacotron2 Model...\n')
        self.tts_model = Tacotron2().to(device)
        # tts_load_path = self.args.tts_weights if self.args.tts_weights else self.paths.tts_latest_weights
        tts_load_path = self.args.tts_weights
        self.tts_model.load(tts_load_path)

        # === Display Conclusion / Information === #
        if self.args.vocoder == 'wavernn':
            self.voc_k = self.voc_model.get_step() // 1000
            self.tts_k = self.tts_model.get_step() // 1000

            simple_table([('Tacotron2', str(self.tts_k) + 'k'),
                          ('Vocoder Type', 'WaveRNN'),
                          ('WaveRNN', str(self.voc_k) + 'k'),
                          ('Generation Mode',
                           'Batched' if self.args.batched else 'Unbatched'),
                          ('Target Samples',
                           self.args.target if self.args.batched else 'N/A'),
                          ('Overlap Samples', self.args.overlap if self.args.batched else 'N/A')])

        elif self.args.vocoder == 'griffinlim':
            self.tts_k = self.tts_model.get_step() // 1000
            simple_table([('Tacotron2', str(self.tts_k) + 'k'),
                          ('Vocoder Type', 'Griffin-Lim'),
                          ('GL Iters', self.args.iters)])

    def gen_tacotron2(self, inputs):
        for i, x in enumerate(inputs, 1):
            print(f'\n| Generating {i}/{len(inputs)}')
            print(x)

            x = np.array(x)[None, :]
            x = torch.autograd.Variable(torch.from_numpy(x)).cuda().long()

            self.tts_model.eval()
            _, mel_outputs_postnet, _, _ = self.tts_model.inference(x)
            if mel_outputs_postnet.shape[2] > 2000:
                print(mel_outputs_postnet.shape)
                # too long, not successful
                return False

            if self.args.vocoder == 'griffinlim':
                v_type = self.args.vocoder
            elif self.args.vocoder == 'wavernn' and self.args.batched:
                v_type = 'wavernn_batched'
            else:
                v_type = 'wavernn_unbatched'

            # === output === #
            # if not self.args.save_dir:
            #     save_path = self.paths.tts_output / \
            #         f'{i}_{v_type}_{self.tts_k}k.wav'
            # else:
            os.makedirs(self.args.save_dir, exist_ok=True)
            save_path = os.path.join(
                self.args.save_dir, f'{i}_{v_type}_{self.tts_k}k.wav')

            if self.args.vocoder == 'wavernn':
                m = mel_outputs_postnet
                wav = self.voc_model.generate(
                    m, self.args.batched, hp.voc_target, hp.voc_overlap, hp.mu_law)
                save_wav(wav, save_path)

            elif self.args.vocoder == 'griffinlim':
                m = torch.squeeze(mel_outputs_postnet).detach().cpu().numpy()
                wav = reconstruct_waveform(m, n_iter=self.args.iters)
                save_wav(wav, save_path)
            # return True

    def generate(self, input_text=None, file=None):
        # generate wavs from a given file
        if file is not None:
            with open(file) as f:
                inputs = [text_to_sequence(
                    l.strip(), hp.text_cleaners) for l in f]
        else:
            inputs = [text_to_sequence(input_text.strip(), ['basic_cleaners'])]
        self.gen_tacotron2(inputs)

        # below is for "Zenbo demo"
        # generate one wav from a given text input
        # else:
        #     inputs = [text_to_sequence(input_text.strip(), ['basic_cleaners'])]
        # success = self.gen_tacotron2(inputs)

        # if not success:
        #     print("TOO LONG!!!")
        #     _input = [text_to_sequence(
        #         'sit8 le1 tsit8 ku2 gua1 be3 hiau1 koŋ2 .', ['basic_cleaners'])]
        #     self.gen_tacotron2(_input)

        print('\n\nDone.\n')


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='TTS')
    parser.add_argument('input_file', type=str,
                        help='[string/path] Input file containing input sentences')

    parser.add_argument('--tts_weights', metavar='tts_weights_dir', type=str,
                        help='[string/path] Load in different Tacotron weights', default=None, required=True)

    parser.add_argument('--voc_weights', metavar='voc_weights_dir', type=str,
                        help='[string/path] Load in different WaveRNN weights', default=None, required=True)
    parser.add_argument('--save_dir', metavar='speech_save_dir', type=str, default="./result")

    args = parser.parse_args()
    args.vocoder = 'wavernn'
    args.hp_file = 'hparams.py'
    args.save_attn = False
    args.batched = True
    args.target = None
    args.overlap = None
    args.force_cpu = False

    input_file = args.input_file
    TTS = TaiwaneseTacotron(args)
    TTS.generate(file=input_file)
