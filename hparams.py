# === Config === #
voc_model = 'wavernn'
tts_model = 'tacotron2'

voc_model_id = 'WaveRNN_A'
tts_model_id = 'Tacotron2_A'

is_cuda = True
pin_mem = True
p = 10  # mel spec loss penalty

# ====== DSP ====== #
sample_rate = 22050
n_fft = 1024
fft_bins = n_fft // 2 + 1
num_mels = 80
hop_length = 275
win_length = 1024
fmin = 40
min_level_db = -100  # wavernn config
ref_level_db = 20
mu_law = True
peak_norm = False


################################
#  WAVERNN / VOCODER           #
################################

if voc_model == 'wavernn':
    # either 'RAW' (softmax on raw bits) or 'MOL' (sample from mixture of logistics)
    # 2020.12.06: guess: for small dataset, don't use MOL. It's hard to converge.
    voc_mode = 'RAW'
    # NB - this needs to correctly factorise hop_length
    voc_upsample_factors = (5, 5, 11)
    voc_rnn_dims = 512
    voc_fc_dims = 512
    voc_compute_dims = 128
    voc_res_out_dims = 128
    voc_res_blocks = 10

    if voc_mode == 'MOL':
        bits = 16
    else:
        bits = 9

    # Training
    voc_batch_size = 32
    voc_lr = 5e-5
    voc_checkpoint_every = 1000

    # number of samples to generate at each checkpoint
    voc_gen_at_checkpoint = 5
    voc_total_steps = 1_000_000         # Total number of training steps
    voc_test_samples = 50               # How many unseen samples to put aside for testing

    # this will pad the input so that the resnet can 'see' wider than input length
    voc_pad = 2
    voc_seq_len = hop_length * 5        # must be a multiple of hop_length
    voc_clip_grad_norm = 4              # set to None if no gradient clipping needed

    # Generating / Synthesizing
    # very fast (realtime+) single utterance batched generation
    voc_gen_batched = True
    # target number of samples to be generated in each batch entry
    voc_target = 11_000
    voc_overlap = 550                   # number of samples for crossfading between batches

    # TACOTRON/TTS -----------------------------------------------------------------------------------------------------#
    ignore_tts = False



################################
# Tacotron 2 / TTS             #
################################

# if you have a couple of extremely long spectrograms you might want to use this
tts_max_mel_len = 1250
# bins the spectrogram lengths before sampling in data loader - speeds up training
tts_bin_lengths = True
# clips the gradient norm to prevent explosion - set to None if not needed
tts_clip_grad_norm = 1.0
tts_checkpoint_every = 1000        # checkpoints the model every X steps

if tts_model == 'tacotron2':
    n_mel_channels = num_mels
    fp16_run = False
    mask_padding = True
    distributed_run = False
    cudnn_enabled = True
    cudnn_benchmark = False
    ignore_layers = ['embedding.weight']
    dynamic_loss_scaling = True
    iters_per_checkpoint = 1000

    from utils.text import symbols
    n_symbols = len(symbols)  # in model config

    # symbols_embedding_dim=512
    symbols_embedding_dim = 256

    # ====== Encoder ====== #
    encoder_kernel_size = 5
    encoder_n_convolutions = 3
    encoder_embedding_dim = 256
    # encoder_embedding_dim=512

    # ====== Decoder ====== #
    n_frames_per_step = 3  # currently only 1 is supported
    decoder_rnn_dim = 512  # 1024
    prenet_dim = 64        # 256
    max_decoder_steps = 1000
    gate_threshold = 0.5
    p_attention_dropout = 0.1
    p_decoder_dropout = 0.1

    # ====== Attention ====== #
    attention_rnn_dim = 512  # 1024
    attention_dim = 128
    # Location Layer parameters
    attention_location_n_filters = 32
    attention_location_kernel_size = 31

    # ====== Post-net ====== #
    # Mel-post processing network parameters
    postnet_embedding_dim = 512
    postnet_kernel_size = 5
    postnet_n_convolutions = 5

    tts_schedule = [(1e-3,  10_000,  16),   # progressive training schedule
                    (1e-3, 100_000,  16),   # (lr, step, batch_size)
                    (1e-4, 180_000,  16),
                    (1e-4, 350_000,  8)]
    # weight_decay=1e-6

    text_cleaners = ['basic_cleaners']
