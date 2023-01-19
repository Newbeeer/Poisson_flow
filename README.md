# Poisson Flow Generative Models For Audio Generation

Based on the Pytorch implementation of the NeurIPS 2022
paper [Poisson Flow Generative Models](https://arxiv.org/abs/2209.11178),

by [Yilun Xu](http://yilun-xu.com)\*, [Ziming Liu](https://kindxiaoming.github.io/#pub)
\*, [Max Tegmark](https://space.mit.edu/home/tegmark/), [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/)

## Dependencies

Conda (Python 3.10.8, CUDA Version 11.3)

```sh
conda env create -f environment.yml

conda activate poisson
```

## Usage

Train our models through `main.py`.

```sh
python3 train.py:
  --conf: <128_deep|64_deep|sd_128|sd_64>: Training configuration.
  --eval_folder: The folder name for storing evaluation results
    (default: 'eval')
  --workdir: Working directory
  --test: Test configuration with small batchsize and verbose output
```

Evaluate our models through `eval.py`.

```sh
python3 eval.py 
  --conf: <128_deep|64_deep|sd_128|sd_64>: Training configuration.
  --workdir: Working directory 
  --checkpoint_dir: Checkpoint directory 
  --ckpt_number: Checkpoint to evaluate
  --sampling: Generate new samples
  --save_audio: Save audio when generating samples
  --enable_benchmarking: Run benchmarking during sampling
```

For example run:

```sh
python eval.py --conf 128_deep --workdir 128_deep  --checkpoint_dir checkpoints/pfgm/128 --ckpt_number 500000 --sampling --save_audio --enable_benchmarking
```

To evaluate and generate samples from the 128 deep network from checkpoint number 500'000.

* `workdir` is the path that stores all artifacts of one experiment, like checkpoints, samples, and evaluation results.

* `eval_folder` is the name of a subfolder in `workdir` that stores all artifacts of the evaluation process, like meta
  checkpoints for pre-emption prevention, image samples, and numpy dumps of quantitative results.

## Checkpoints

Please place the pretrained checkpoints under the directory `workdir/checkpoints`, e.g., `cifar10_ddpmpp/checkpoints`.

## Mean Opinion Score

Please find the code used for conducting the mean opinion score survey in the link below:

https://github.com/TSHP/PFGM_MOS

## Google Drive

Please find our dataset, the checkpoints of the models and the default configuration used to obtain the results as
described in our report in the Google drive below.

https://drive.google.com/drive/folders/18cMaVX4o5fwFsZ8ZPpnvWDg9H57EaX3_?usp=share_link

