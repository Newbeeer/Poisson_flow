# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training and evaluation"""
# collections fix for python 10
import collections
import collections.abc

collections.Container = collections.abc.Container
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

import logging
import run_lib
import os
import argparse
from configs.get_configs import get_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--eval_folder", default="eval")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    args.config = get_config(args)

    if args.mode == "train":
        print("START TRAINING")
        # Create the working directory
        os.makedirs(args.workdir, exist_ok=True)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        gfile_stream = open(os.path.join(args.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
        # Run the training pipeline
        run_lib.train(args.config, args.workdir)
    elif args.mode == "eval":
        print("START EVALUATION")
        pass
        # Run the evaluation pipeline
        run_lib.evaluate(args.config, args.workdir, args.eval_folder)
    else:
        raise ValueError(f"Mode {args.mode} not recognized.")


if __name__ == "__main__":
    main()
