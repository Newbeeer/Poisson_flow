# local dictionary holding configs
_CONFS = {}


def register_config(cls=None, *, name=None):
    """A decorator for registering configs."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CONFS:
            raise ValueError(f'Already registered config with name: {local_name}')
        _CONFS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_config(args):
    config =  _CONFS[args.conf]()

    # set sizes for test mode
    if hasattr(args, 'test'):
        if args.test:
            config.training.batch_size = 2
            config.eval.batch_size = 1
            config.training.small_batch_size = 2
            config.training.accum_iter = 1
            config.training.eval_freq = 1
            config.training.snapshot_freq = 1
            config.training.snapshot_freq_for_preemption = 1

    print("Read Config: ", config, sep='\n')

    return config
