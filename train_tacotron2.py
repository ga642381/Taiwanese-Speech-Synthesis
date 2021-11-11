import torch
from torch import optim
import torch.nn.functional as F
from utils import hparams as hp
from utils.display import *
from utils.text.symbols import symbols
from utils.paths import Paths

import argparse
from utils import data_parallel_workaround
import os
from pathlib import Path
import time
import numpy as np
import sys
from utils.checkpoints import save_checkpoint, restore_checkpoint
from utils.logger import Tacotron2Logger
from utils.dataset import get_tacotron2_datasets
from tacotron2 import Tacotron2
from tacotron2 import Tacotron2Loss


def np_now(x: torch.Tensor): return x.detach().cpu().numpy()


def main(args):
    hp.configure(args.hp_file)  # Load hparams from file
    paths = Paths(args.data_dir, hp.voc_model_id, hp.tts_model_id)

    force_train = args.force_train
    force_gta = args.force_gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        for session in hp.tts_schedule:
            #_, _, _, batch_size = session
            _, _, batch_size = session
            if batch_size % torch.cuda.device_count() != 0:
                raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    # Instantiate Tacotron Model
    print('\nInitializing Tacotron2 Model...\n')
    model = Tacotron2().to(device)
    model.num_params()

    logger = Tacotron2Logger(paths.tts_tensorboard)

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('tts', paths, model, optimizer, create_if_missing=True)
    criterion = Tacotron2Loss()

    if not force_gta:
        for i, session in enumerate(hp.tts_schedule):
            current_step = model.get_step()
            # lr, max_step, batch_size = session
            lr, max_step, batch_size = session
            training_steps = max_step - current_step
            
            # ===
            # Do we need to change to the next session?
            if current_step >= max_step:
                # Are there no further sessions than the current one?
                if i == len(hp.tts_schedule)-1:
                    # There are no more sessions. Check if we force training.
                    if force_train:
                        # Don't finish the loop - train forever
                        training_steps = 999_999_999
                    else:
                        # We have completed training. Breaking is same as continue
                        break
                else:
                    # There is a following session, go to it
                    continue
            # ===
            
            simple_table([('Training Steps', str(training_steps//1000) + 'k Steps'),
                            ('Batch Size', batch_size),
                            ('Learning Rate', lr)])
            
            print(paths.data)
            train_set, attn_example = get_tacotron2_datasets(paths.data, batch_size)
            # === !!train!! === #
            tts_train_loop(paths, model, optimizer, criterion, logger, train_set, lr, training_steps, attn_example)

        print('Training Complete.')
        print('To continue training increase tts_total_steps in hparams.py or use --force_train\n')

    print('Creating Ground Truth Aligned Dataset...\n')
    train_set, attn_example = get_tacotron2_datasets(paths.data, 8)
    create_gta_features(model, train_set, paths.gta)

    print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')


def tts_train_loop(paths: Paths, model: Tacotron2, optimizer, criterion, logger, train_set, lr, train_steps, attn_example):
    
    device = next(model.parameters()).device  # use same device as model parameters
    for g in optimizer.param_groups: g['lr'] = lr
    total_iters = len(train_set)
    epochs = train_steps // total_iters + 1
    model.train()
    
    for e in range(1, epochs+1):
        start = time.time()
        running_loss = 0

        # Perform 1 epoch
        for i, batch in enumerate(train_set, 1):

            
            #
            item_ids, x, y = model.parse_batch(batch)
            y_pred = model(x)
            
            # loss
            loss, item = criterion(y_pred, y)
            
            # zero grad
            model.zero_grad()
            loss.backward()
            
            if hp.tts_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.tts_clip_grad_norm)
                if np.isnan(grad_norm.cpu()):
                    print('grad_norm was NaN!')

            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            if step % hp.tts_checkpoint_every == 0:
                ckpt_name = f'taco_step{k}K'
                save_checkpoint('tts', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)

            if attn_example in item_ids:
                _, mel_out_postnet, _, alignments = y_pred
                attention = alignments 
                
                idx = item_ids.index(attn_example)

                logger.log(y, y_pred, idx, step)
                save_attention(np_now(attention[idx]), paths.tts_attention/f'{step}')
                save_spectrogram(np_now(mel_out_postnet[idx]), paths.tts_mel_plot/f'{step}', 600)

            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:#.4} | {speed:#.2} steps/s | Step: {k}k | '
            stream(msg)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('tts', paths, model, optimizer, is_silent=True)
        model.log(paths.tts_log, msg)
        print(' ')


def create_gta_features(model: Tacotron2, train_set, save_path: Path):
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set)
    
    model.eval()
    for i, batch in enumerate(train_set, 1):
        #  item_ids = (item_ids),
        #         x = (text_padded, input_lengths, mel_padded, max_len, output_lengths),
        #         y = (mel_padded, gate_padded)
        item_ids, x, _ = model.parse_batch(batch)
        _, _, _, _ , output_lengths = x
        with torch.no_grad():
            y_pred = model(x)
            
        mel_out, mel_out_postnet, gate_out, _ = y_pred
        gta = mel_out_postnet
        gta = gta.cpu().numpy()
        for j, item_id in enumerate(item_ids):
            mel = gta[j][:, :output_lengths[j]]
            np.save(save_path/f'{item_id}.npy', mel, allow_pickle=False)

        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron2 TTS')
    parser.add_argument('data_dir', help='Relative path of dataset.pkl')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()
    #args.force_gta = True
    main(args)
