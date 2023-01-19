import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()
    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 256  # bs to calculate the gt field
    training.n_iters = 500000  # 100k takes 17 hours on 4 gpus rtx 6000
    training.snapshot_freq = 10000
    training.log_freq = 100
    training.eval_freq = 10000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 1000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.reduce_mean = False
    training.M = 280
    training.amp = False
    training.accum_iter = 0

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = True
    sampling.probability_flow = False
    sampling.snr = 0.2
    sampling.N = 1
    sampling.z_exp = 1
    sampling.rk_stepsize = 0.9

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.batch_size = 8
    evaluate.enable_sampling = True
    evaluate.num_samples = 32
    evaluate.enable_loss = False
    evaluate.save_images = True  # debugging
    evaluate.show_sampling = False
    
    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = 'speech_commands'
    # audio related things
    data.centered = False
    data.num_channels = 1
    data.add_noise = False
    
    # model
    config.model = model = ml_collections.ConfigDict()
    model.sigma_max = 378
    model.sigma_min = 0.01
    model.num_scales = 2000
    model.beta_min = 0.1
    model.beta_max = 20.
    model.dropout = 0.
    model.embedding_type = 'fourier'

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.
    optim.scheduler = 'none'  # 'none', 'OneCylce'
    optim.T_max = 2000  # the period in STEPS (check the total steps for good idea)
    optim.max_lr = 3e-4
    config.seed = 49
    config.device = "cuda" if torch.cuda.is_available() else torch.device('cpu')

    return config


def get_mels_64():
    spec = ml_collections.ConfigDict()
    spec.num_mels = 64
    spec.nfft = 1024
    spec.hop_length = 256
    spec.sample_rate = 16_000
    spec.fmin = 20
    spec.audio_length = 1
    spec.image_size = spec.audio_length * spec.sample_rate // spec.hop_length + 2  # this is 64 which fits the num mels
    spec.spec_len_samples = spec.image_size
    return spec


def get_mels_128():
    spec = ml_collections.ConfigDict()
    spec.num_mels = 128
    spec.nfft = 512
    spec.hop_length = 128
    spec.sample_rate = 16_000
    spec.fmin = 20
    spec.audio_length = 1
    spec.image_size = spec.audio_length * spec.sample_rate // spec.hop_length + 3  # this is 128 which fits the num mels
    spec.spec_len_samples = spec.image_size
    return spec
