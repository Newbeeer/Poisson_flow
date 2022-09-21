# Poisson Flow Generative Models

Pytorch implementation of the paper [Poisson Flow Generative Models](https://openreview.net/forum?id=voV_TRqcWh&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DNeurIPS.cc%2F2022%2FConference%2FAuthors%23your-submissions)), 

By Yilun Xu\*, Ziming Liu\*, Max Tegmark,Tommi S. Jaakkola



We propose a new **Poisson flow** generative model~(PFGM) that maps a uniform distribution on a high-dimensional hemisphere into any data distribution. We interpret the data points as electrical charges on the $z=0$ hyperplane in a space augmented with an additional dimension $z$, generating a high-dimensional electric field (the gradient of the solution to Poisson equation). We prove that if these charges flow upward along electric field lines, their initial distribution in the $z=0$ plane transforms into a distribution on the hemisphere of radius $r$ that becomes *uniform* in the $r \to\infty$ limit. To learn the bijective transformation, we estimate the normalized field {in the augmented space}. For sampling, we devise a backward ODE that is anchored by the physically meaningful additional dimension: the samples hit the (unaugmented) data manifold when the $z$ reaches zero. 

![schematic](assets/combine.png)

Experimentally, PFGM achieves **current state-of-the-art** performance among the normalizing flow models on CIFAR-10, with an Inception score of **9.68** and a FID score of **2.48**. It also performs on par with the state-of-the-art SDE approaches while offering **10x **to **20x** acceleration on image generation tasks. Additionally, PFGM appears more tolerant of estimation errors on a weaker network architecture and robust to the step size in the Euler method, and capable of scale-up to higher resolution datasets.

---



*Acknowledgement:* Our implementation heavily relies on the repo https://github.com/yang-song/score_sde_pytorch. 

### Dependencies

Run the following to install a subset of necessary python packages for our code

```sh
pip install -r requirements.txt
```

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

Please place the pretrained checkpoints under the directory `workdir/checkpoints`, e.g., `cifar10_ddpmpp/workdir`.  To evaluate the sample, you could execute:

```shell
python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval --workdir cifar10_ddpmpp --config.eval.enable_sampling
```

All checkpoints are provided in this [Google drive folder](https://drive.google.com/drive/folders/1v4u0OhZ0rxjgch51pZLySztMQATQQOeK?usp=sharing).

| Checkpoint path                                              |    Invertible?     |  IS  |  FID  | NFE  |
| :----------------------------------------------------------- | :----------------: | :--: | :---: | :--: |
| [`poisson/cifar10_ddpmpp/`](https://drive.google.com/drive/folders/1UBRMPrABFoho4_laa4VZW733RJ0H_TI0?usp=sharing) | :heavy_check_mark: | 9.62 | 2.54  | ~110 |
| [`poisson/cifar10_ddpmpp_deep/`](https://drive.google.com/drive/folders/1mM-VjygRzF2Z_v4MapdDcZZaUyLy3QOU?usp=sharing) | :heavy_check_mark: | 9.68 | 2.48  | ~110 |
| [`poisson/bedroom_ddpmpp/`](https://drive.google.com/drive/folders/1uFmlcBTQmUI_ZfyUiYoR54H4V2uBsuS7?usp=sharing) | :heavy_check_mark: |  -   | 13.66 | ~122 |



### Tips

TODO