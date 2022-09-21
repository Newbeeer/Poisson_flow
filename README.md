# Poisson Flow Generative Models



![schematic](assets/combine.png)

Our implementation heavily relies on the repo https://github.com/yang-song/score_sde_pytorch. 

### Usage

Train and evaluate our models through `main.py`.

```sh
python3 main.py:
  --config: Training configuration.
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --mode: <train|eval>: Running mode: train or eval
  --workdir: Working directory
```

For example, to train a new PFGM w/ DDPM++ model on CIFAR-10 dataset, one could execute 

```sh
python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode train --workdir poisson_ddpmpp
```



* `config` is the path to the config file. The prescribed config files are provided in `configs/`. They are formatted according to [`ml_collections`](https://github.com/google/ml_collections) and should be quite self-explanatory.

  **Naming conventions of config files**: the path of a config file is a combination of the following dimensions:

  - Method: One of `poisson`, `ve`, `vp`, `sub_vp`

  *  dataset: One of `cifar10`, `celeba`, `celebahq`, `celebahq_256`, `ffhq_256`, `celebahq`, `ffhq`.
  * model: One of `ncsn`, `ncsnv2`, `ncsnpp`, `ddpm`, `ddpmpp`.
  * continuous: train the model with continuously sampled time steps. 

*  `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

* `mode` is either "train" or "eval". When set to "train", it starts the training of a new model, or resumes the training of an old model if its meta-checkpoints (for resuming running after pre-emption in a cloud environment) exist in `workdir/checkpoints-meta` .

* Below are the list of evalutation command-line flags:

  `--config.eval.enable_sampling`: Generate samples and evaluate sample quality, measured by FID and Inception score. 

   `--config.eval.enable_bpd` : Compute log-likelihoods

   `--config.eval.dataset=train/test` : Indicate whether to compute the likelihoods on the training or test dataset.

   `--config.eval.enable_interpolate` : Image Interpolation

   `--config.eval.enable_rescale` : Temperature scaling



### Checkpoints

