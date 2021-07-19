import os
from pathlib import Path


class Paths:
    """Manages and configures the paths used by WaveRNN, Tacotron, and the data."""

    def __init__(self, data_path, voc_id, tts_id):
        self.base = Path(__file__).parent.parent.expanduser().resolve()

        # === 1.Data Paths === #
        self.data = Path(data_path).expanduser().resolve()
        self.quant = self.data/'quant'
        self.mel = self.data/'mel'
        self.gta = self.data/'gta'

        # === 2.WaveRNN / Vocoder Paths === #
        # base
        self.voc_record = self.base/'training_record'/f'{voc_id}.wavernn'
        # checkpoints
        self.voc_checkpoints = self.voc_record/'checkpoints'
        self.voc_latest_weights = self.voc_checkpoints/'latest_weights.pyt'
        self.voc_latest_optim = self.voc_checkpoints/'latest_optim.pyt'
        # output_samples
        self.voc_output = self.voc_record/'samples'/f'{voc_id}.wavernn'
        # training log
        self.voc_log = self.voc_record/'log.txt'
        self.voc_tensorboard = self.voc_record/'tensorboard_log'
        # others
        self.voc_step = self.voc_record/'step.npy'

        # === 3.Tactron2 / TTS Paths ===#
        # base
        self.tts_record = self.base/'training_record'/f'{tts_id}.tacotron2'
        # checkpoints
        self.tts_checkpoints = self.tts_record/'checkpoints'
        self.tts_latest_weights = self.tts_checkpoints/'latest_weights.pyt'
        self.tts_latest_optim = self.tts_checkpoints/'latest_optim.pyt'
        # output_samples
        self.tts_output = self.tts_record/'samples'/f'{tts_id}.tacotron2'
        # training log
        self.tts_log = self.tts_record/'log.txt'
        self.tts_tensorboard = self.tts_record/'tensorboard_log'
        # others
        self.tts_step = self.tts_record/'step.npy'
        self.tts_attention = self.tts_record/'attention'
        self.tts_mel_plot = self.tts_record/'mel_plots'

        self.create_paths(output_only=output_stage)

    def create_paths(self):
        # data
        os.makedirs(self.data, exist_ok=True)
        os.makedirs(self.quant, exist_ok=True)
        os.makedirs(self.mel, exist_ok=True)
        os.makedirs(self.gta, exist_ok=True)

        # record
        os.makedirs(self.voc_record, exist_ok=True)
        os.makedirs(self.voc_checkpoints, exist_ok=True)
        os.makedirs(self.voc_tensorboard, exist_ok=True)
        os.makedirs(self.voc_output, exist_ok=True)

        os.makedirs(self.tts_record, exist_ok=True)
        os.makedirs(self.tts_checkpoints, exist_ok=True)
        os.makedirs(self.tts_tensorboard, exist_ok=True)
        os.makedirs(self.tts_attention, exist_ok=True)
        os.makedirs(self.tts_mel_plot, exist_ok=True)
        os.makedirs(self.tts_output, exist_ok=True)

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
