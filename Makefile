SHELL := /bin/bash
gpus = ""
name = ""

# IMPORTANT!
# You need one environment variable called ENV_FILE that should be set to the 
# your environment file like so:
# export ENV_FILE=setup/environment/default.sh
# This gets read by the Makefile before running any commands.
env_file = $(ENV_FILE)

# Install Poetry in the recommended way.
poetry:
	curl -sSL https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py | python

# Install all the requirements into the conda environment after activating it.
install: poetry
	poetry update
	poetry install

# If you add or delete requirements via poetry add, make sure to run 
# make freeze_requirements to update the requirements file.
freeze_requirements:
	poetry export -f requirements.txt --without-hashes > docker_requirements.txt

# Update the underlying Docker image if necessary.
docker:
	make freeze_requirements
	source $(env_file) && \
	bash ./setup/build_docker_image.sh

# Check to make sure your environment variables are set up correctly.
check_environment:
	echo $(env_file)
	source $(env_file) && \
	printenv

# Check to make sure the container has the same environent variables.
check_container_environment:
	echo $(env_file)
	source $(env_file) && \
	make run_in_container command=printenv

# Helper for running stuff in the container, masking gpus via a string (e.g. gpus="0,1").
# Also needs a name argument for what to name the container. Defaults to "", which means
# the name will be created by the DockerRunner class using the command.
# If run_in_container fails, make run_in_host will be tried with the same arguments, 
# except NAME.
run_in_container:
	-docker info >/dev/null 2>&1 && \
	source $(env_file) && \
	CUDA_VISIBLE_DEVICES=$(gpus) NAME=$(name) python -m runners.run_in_docker "$(command)" || \
	(echo "Trying on host..." && make run_in_host command="$(command)" env_file=$(env_file) gpus=$(gpus))

# Helper for running stuff on the host, masking gpus via a string (e.g. gpus="0,1").
run_in_host:
	source $(env_file) && \
	CUDA_VISIBLE_DEVICES=$(gpus) $(command)

# Start a Jupyter notebook in a container. It'll be available at the host port defined 
# in your environment setup.
jupyter:
	make command="jupyter lab --ip=0.0.0.0" gpus=$(gpus) name=$(name) run_in_container

# Start a Tensorboard server in a container. It'll be available at the host port defined 
# in your environment setup.
tensorboard:
	source $(env_file) && \
	make command="tensorboard --logdir $(ARTIFACTS_DIRECTORY) --bind_all" name=tensorboard run_in_container

# Remove temporary files created by tests.
clean:
	rm -rf tests/out/

# Run all the test cases.
test:
	make test_host args=$(args)

# Run all the test cases in the host.
test_host:
	make gpus=$(gpus) command="python -m pytest --cov=. . -v $(args)" run_in_host

# Run all the test cases in the container.
test_container:
	make gpus=$(gpus) command="python -m pytest --cov=. . -v" run_in_container

# Run a whole pipeline file correctly.
# Usage:
#	make pipeline yml=path/to/pipeline.yml
pipeline:
	source $(env_file) && \
	python -m scripts.pipeline -y "${yml}"

experiment:
	source $(env_file) && \
	python -m scripts.sweep_experiment -p $(yml) --num_gpus $(num_gpus) --num_jobs $(num_jobs) 

diverse_experiments:
	source $(env_file) && \
	python -m scripts.diverse_experiments -y $(yml)