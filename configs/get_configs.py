from .audio_ddpmpp_128_deep import get_config as ddpmpp_128_deep
from .audio_ddpmpp_64_deep import get_config as ddpmpp_64_deep
from .audio_sd_128 import get_config as sd_128
from .audio_sd_64 import get_config as sd_64
from .audio_diffwave_128 import get_config as dw_128

def get_config(args):
    if args.conf == "128_deep":
        config = ddpmpp_128_deep()
    elif args.conf == "64_deep":
        config = ddpmpp_64_deep()
    elif args.conf == "sd_128":
        config = sd_128()
    elif args.conf == "sd_64":
        config = sd_64()
    elif args.conf == "dw_128":
        config = dw_128()
    else:
        raise ValueError("Unknown conf name!")

    # set sizes for test mode
    if args.test:
        config.training.continuous = True
        config.training.batch_size = 4
        config.eval.batch_size = 4
        config.training.small_batch_size = 4
        config.training.eval_freq = 100
        config.training.snapshot_freq = 100
        config.training.snapshot_freq_for_preemption = 250

    print("Read Config: ", config, sep='\n')

    return config
