# Training + inference image for the PolyUMI visuomotor diffusion policy.
#
# Reproduces UMI's conda environment (conda_environment.yaml) via micromamba, so no host conda
# is needed — which is the whole point (conda fights the ROS install on the laptop). One image
# serves both entrypoints: docker/train.sh (default) and docker/serve.sh (inference server),
# so the serving env is byte-identical to the training env (checkpoints are dill-pickled and
# must unpickle against the same dep tree).
#
# Build + run instructions, including the rootless-Docker flags, live in the PolyUMI repo at
# docs/training-instructions.md. Build on the GPU workstation (rootless, no registry push).

ARG MICROMAMBA_VERSION=1.5.8
FROM mambaorg/micromamba:${MICROMAMBA_VERSION}

# System libraries required by the eval-only pip deps in conda_environment.yaml:
#   libspnav-dev                                   -> spnav (spacemouse teleop)
#   libosmesa6-dev libgl1-mesa-glx libglfw3 patchelf -> free-mujoco-py / robosuite
# These are NOT needed for training (our env_runner is the no-op RealPushTImageRunner). When the
# fork is pruned (PolyUMI plan Step 4) they can be removed alongside those deps. apt runs as
# build-time root even under rootless Docker, so no host sudo is involved.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libspnav-dev \
        libosmesa6-dev \
        libgl1-mesa-glx \
        libglfw3 \
        patchelf \
    && rm -rf /var/lib/apt/lists/*
USER $MAMBA_USER

# Create the 'umi' env from the upstream spec, left unmodified so it stays a clean mirror.
COPY --chown=$MAMBA_USER:$MAMBA_USER conda_environment.yaml /tmp/conda_environment.yaml
RUN micromamba env create -y -f /tmp/conda_environment.yaml \
    && micromamba clean --all --yes

# Server deps layered on top rather than added to the yaml, so the fork's conda_environment.yaml
# keeps mirroring UMI exactly. fastapi/uvicorn are tiny and don't perturb the DP dep tree.
RUN micromamba run -n umi pip install --no-cache-dir \
        "fastapi>=0.115" \
        "uvicorn[standard]>=0.32"

# Project code. The dataset and output/checkpoint dirs are bind-mounted at run time (see
# .dockerignore and the run wrapper), not copied — native I/O speed, no image bloat.
COPY --chown=$MAMBA_USER:$MAMBA_USER . /app
WORKDIR /app

# Activate 'umi' for subsequent RUNs and for the container's process, so `python` is the env's.
ARG MAMBA_DOCKERFILE_ACTIVATE=1
ENV ENV_NAME=umi

# Default entrypoint trains; override the command (docker/serve.sh) to run the inference server.
# The base image's _entrypoint.sh activates ENV_NAME before exec'ing the command.
CMD ["bash", "docker/train.sh"]
