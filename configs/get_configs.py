from .audio_ddpmpp_128_deep import get_config as ddpmpp_128_deep
from .audio_ddpmpp_64_deep import get_config as ddpmpp_64_deep
from .audio_sd import get_config as sd

def get_config(args):
    if args.conf == "128_deep":
        config =  ddpmpp_128_deep()
    elif args.conf == "64_deep":
        config = ddpmpp_64_deep()
    elif args.conf == "sd":
        config = sd()
        
    # set sizes for test mode
    if args.test:
        config.training.continuous = True
        config.training.batch_size = 2
        config.training.small_batch_size = 2
        config.training.eval_freq = 100
        config.training.snapshot_freq = 100
        config.training.snapshot_freq_for_preemption = 200
    
    return config