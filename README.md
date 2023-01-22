# Poisson Flow Generative Models For Audio Generation

Based on the Pytorch implementation of the NeurIPS 2022
paper [Poisson Flow Generative Models](https://arxiv.org/abs/2209.11178),

by [Yilun Xu](http://yilun-xu.com)\*, [Ziming Liu](https://kindxiaoming.github.io/#pub)
\*, [Max Tegmark](https://space.mit.edu/home/tegmark/), [Tommi S. Jaakkola](http://people.csail.mit.edu/tommi/)

![](/assets/sampling.gif)

## Dependencies

Tested on Ubuntu with Python 3.10

```sh
conda create -n poisson python=3.10 --yes
conda activate poisson
pip install -r requirements.txt
```

## Usage
To train a Mel PFGM (MPFGM) you need to generate a dataset and specify a configuration file that defines the model, the data and sampling methods.

### Config File
You can find some default configuration files in the `configs` folder. There are many parameters to be set. It is important to set the right path for `data.mel_root`. This is the location of the Mel spectrogram dataset. If you want to define your own one, use the same structure, register it via decorator and add it to the imports in `generate_dataset.py`, `train.py` and `eval.py`.

### Dataset
To generate a dataset you should place your audio data such that each class is in a sepperate folder. For the SpeechCommands Dataset using only classes of digits from zero to nine the folder looks as follows:
```
mel_sc09_128
├── eight
├── five
├── four
├── nine
├── one
├── seven
├── six
├── three
├── two
└── zero
```
To convert the dataset run
```sh
python3 generate_dataset.py
  --input_dir: dataset root
  --target_dir: mels are saved here
  --conf: name of the registered config file
```

Finally you have to calculate the dataset related hyperparameters using the `hyperparameter_calculator.py` script with

```sh
python3 hyperparam_calculator.py --conf <name of registered config file>

```
You need to fill in the `M`, `zmax` and `upper norm` values in the config file.
### Training
Train a model through `train.py`. Don't forget to set the right dataset path, if you generated your own one and to calculate its hyperparameters.

```sh
python3 train.py:
  --conf: <128_deep|64_deep|sd_128|sd_64>: Training configuration.
  --workdir: Working directory
  --test: Test mode with small batchsize and verbose output
  --DDP: Using Distributed Data Parallel
  --wandb: Using weights and biases. Use wandb login beforehand
```
### Evaluation
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

Please place the pretrained checkpoints under the directory `workdir/checkpoints`, e.g., `128_deep/checkpoints`.

## Mean Opinion Score

Please find the code used for conducting the mean opinion score survey in the link below:

https://github.com/TSHP/PFGM_MOS

## Google Drive

Please find our dataset, the checkpoints of the models and the default configuration used to obtain the results as
described in our report in the Google drive below.

https://drive.google.com/drive/folders/18cMaVX4o5fwFsZ8ZPpnvWDg9H57EaX3_?usp=share_link

