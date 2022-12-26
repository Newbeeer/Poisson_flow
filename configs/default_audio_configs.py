import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 256 #bs to calculate the gt field
  training.n_iters = 2400001
  training.snapshot_freq = 5000
  training.log_freq = 50
  training.eval_freq = 5000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 5000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.continuous = True
  training.reduce_mean = False
  training.M = 280

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.1
  sampling.N = 1
  sampling.z_exp = 1

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 0
  evaluate.end_ckpt = 20
  evaluate.batch_size = 32
  evaluate.enable_sampling = True
  evaluate.enable_interpolate = False
  evaluate.num_samples = 100
  evaluate.enable_loss = False
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.save_images = True # debugging
  evaluate.enable_rescale = False

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'speech_commands'
  # audio related things
  data.num_mels = 64
  data.nfft = 1024
  data.hop_length = 256
  data.sample_rate = 16_000 # diffwave uses 22050
  data.audio_length = 1 # length in seconds
  data.image_size = data.audio_length * data.sample_rate // data.hop_length + 2
  data.spec_len_samples = data.image_size

  data.uniform_dequantization = False
  data.centered = False
  data.num_channels = 1

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
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.

  config.seed = 49
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config