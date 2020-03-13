from . import cmd, document_parser
import yaml
from shutil import copyfile
import os
from random import randint
import subprocess
from argparse import ArgumentParser
from time import sleep

"""
This script uses the split datasets from 'song_diversity.py' to create generated mixtures, then subsequently run an experiment with them.
"""

def make_mixtures(training_folder, base_mixture_yml):
    """
    training_folder: filepath to folder containing training data
    base_mixture_yml: filepath to a mixture yml, to be copied and edited to make mixtures of the training data

    returns:
    generated_train, generated_val, generated_test
    """

    # Makes a new yaml
    gen_yml = base_mixture_yml[:base_mixture_yml.rfind("/") + 1] + str(randint(1000, 1000000000)) + ".yml"
    with open(base_mixture_yml) as file:
        yml_dumps = yaml.load(file, Loader=yaml.Loader)
    print(yml_dumps)
    yml_dumps["mixture_parameters"]["train"]["foreground_path"] = training_folder
    yml_dumps["mixture_parameters"]["val"]["foreground_path"] = training_folder
    with open(gen_yml, "w") as file:
        yaml.dump(yml_dumps, file)
    print(f"yaml {gen_yml} has been generated")
    subprocess.run(f"python -m scripts.mix_with_scaper -y {gen_yml}", shell=True)

    os.remove(gen_yml)
    return (yml_dumps["mixture_parameters"]["train"]["target_path"], 
        yml_dumps["mixture_parameters"]["val"]["target_path"], 
        yml_dumps["mixture_parameters"]["test"]["target_path"])


def make_experiment(train_folder, val_folder, test_folder, base_experiment_yml, num_gpus, num_jobs, key=None):
    """
    training_folder: filepath to folder containing training data
    base_experiment_yml: filepath to a experiment yml, to be copied and edited to make experiments with the training data
    returns:
    key
    """
    if key is None:
        key = str(randint(1000, 1000000000))
    gen_yml = base_experiment_yml[:base_experiment_yml.rfind("/") + 1] + key + ".yml"
    with open(base_experiment_yml) as file:
        yml_dumps = yaml.load(file, Loader=yaml.Loader)
    print(yml_dumps)
    yml_dumps["datasets"]["train"]["folder"] = train_folder
    yml_dumps["datasets"]["val"]["folder"] = val_folder
    yml_dumps["datasets"]["test"]["folder"] = test_folder

    with open(gen_yml, "w") as file:
        yaml.dump(yml_dumps, file)

    print(f"yaml {gen_yml} has been generated")
    subprocess.run(f"make experiment yml={gen_yml} num_gpus={num_gpus} num_jobs={num_jobs}", shell=True)
    os.remove(gen_yml)
    return key


def parse_array(string, _type=str):
    if string[0] == "[" and string[-1] == "]":
        string = string[1:-1]
    array = string.split(",")
    for i in range(len(array)):
        array[i] = _type(array[i].strip("\"' "))
    return array

def diverse_experiments(**kwargs):
    training_folders = parse_array(kwargs["training_folders"])
    base_mixture_yml = kwargs["base_mixture_yml"]
    base_experiment_yml = kwargs["base_experiment_yml"]
    num_jobs = kwargs["num_jobs"]
    num_gpus = kwargs["num_gpus"]

    for folder in training_folders:
        generated_train, generated_val, generated_test = make_mixtures(folder, base_mixture_yml)
        key = make_experiment(generated_train, generated_val, generated_test, base_experiment_yml, num_gpus, num_jobs)
        subprocess.run(f"make pipeline yml=experiments/dpcl/out/{key}/pipeline.yml", shell=True)


@document_parser('diverse_experiments', 'scripts.diverse_experiments.diverse_experiments')
def build_parser():
    parser = ArgumentParser()
    parser.add_argument(
        "--training_folders",
        type=str,
        required=True,
        help="""List of training folders file path. For each folder, an experiment will be ran."""
    )
    parser.add_argument(
        "--base_mixture_yml",
        type=str,
        required=True,
        help="""Filepath to an coherent.yml/incoherent.yml in the data prep. It will be copied, and the training dataset will be changed to be one from the training_folder. After doing so, a new training set will be made."""
    )
    parser.add_argument(
        "--base_experiment_yml",
        type=str,
        required=True,
        help="""Filepath to an experiment Yaml. Will be copied and used for generating experiment pipeline"""
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        required=True,
        help="""Filepath to an experiment Yaml. Will be copied and used for generating experiment pipeline"""
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        required=True,
        help="""Filepath to an experiment Yaml. Will be copied and used for generating experiment pipeline"""
    )

    return parser

if __name__ == "__main__":
    cmd(diverse_experiments, build_parser)
