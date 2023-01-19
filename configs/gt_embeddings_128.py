import ml_collections
from configs.default_audio_configs import get_mels_128


def get_config():
    config = ml_collections.ConfigDict()

    config.data = data = ml_collections.ConfigDict()
    data.spec = ml_collections.ConfigDict()
    data.spec = get_mels_128()
    data.image_height = data.spec.image_size
    data.image_width = data.spec.image_size
    data.mel_root = '/cluster/scratch/krasnopk/data/poisson/datasets/SpeechCommands/speech_commands_v0.02_mel'
    data.channels = 1
    data.category = 'mel'  # audio, mel
    data.centered = False
    data.dataset = 'speech_commands'

    config.data_path = "/cluster/scratch/krasnopk/data/poisson/datasets/SpeechCommands/speech_commands_v0.02"

    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 256  # bs to calculate the gt field
    training.n_iters = 500000  # 100k takes 17 hours on 4 gpus rtx 6000
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

    return config
