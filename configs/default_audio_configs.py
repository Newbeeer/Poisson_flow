import ml_collections
import torch
from configs.mel_configs import get_mels_64, get_mels_128

def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 256 #bs to calculate the gt field
  training.n_iters = 500000 # 100k takes 17 hours on 4 gpus rtx 6000
  training.snapshot_freq = 10000
  training.log_freq = 50
  training.eval_freq = 5000
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 500
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.continuous = True
  training.reduce_mean = False
  training.M = 280
  training.amp = False
  training.accum_iter = 0
  
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
  evaluate.begin_ckpt = 30
  evaluate.end_ckpt = 31
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
  optim.optimizer = 'AdamW'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 1000
  optim.grad_clip = 1.
  optim.scheduler = 'none' # 'none', 'OneCylce' 
  optim.T_max = 2000 # the period in STEPS (check the total steps for good idea)
  optim.max_lr = 3e-4
  config.seed = 49
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config