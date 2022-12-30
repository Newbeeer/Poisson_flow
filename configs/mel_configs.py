import ml_collections

def get_mels_64():
    spec = ml_collections.ConfigDict()
    spec.num_mels = 64
    spec.nfft = 1024
    spec.hop_length = 256
    spec.sample_rate = 16_000 # diffwave uses 22050
    spec.audio_length = 1 # length in seconds
    spec.image_size = spec.audio_length * spec.sample_rate // spec.hop_length + 2 # this is 64 which fits the num mels
    spec.spec_len_samples = spec.image_size
    return spec

def get_mels_128():
    spec = ml_collections.ConfigDict()
    spec.num_mels = 128
    spec.nfft = 512
    spec.hop_length = 128
    spec.sample_rate = 16_000 # diffwave uses 22050
    spec.audio_length = 1 # length in seconds
    spec.image_size = spec.audio_length * spec.sample_rate // spec.hop_length + 3 # this is 128 which fits the num mels
    spec.spec_len_samples = spec.image_size
    return spec