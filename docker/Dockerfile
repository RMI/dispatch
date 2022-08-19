FROM condaforge/mambaforge:4.12.0-0

# Create a non-root user inside the container
RUN useradd -Ums /bin/bash catalyst

# Copy the cloned repository into the user's home directory
COPY --chown=catalyst:catalyst . /home/catalyst/cheshire

# Switch to being the catalyst user and go into the copied repo
USER catalyst
WORKDIR /home/catalyst/cheshire

# Create a conda environment based on the specification in the repo
# We need information from .git to get version with setuptools_scm so we mount that
# directory without copying it into the image.
RUN --mount=type=bind,source=.git,target=/home/catalyst/cheshire/.git \
        # mamba create can't read environment.yml, and mamba env isn't installed yet,
        # so we create a bare py3.10 env first:
        mamba create --name cheshire --yes python=3.10 && \
    # Then we can use mamba env update, which can parse the environment.yml file:
    mamba env update --name cheshire --file environment.yml && \
    conda clean -afy && \
    conda run --name cheshire pip install --no-cache-dir -e './[dev,docs,tests]'

# Run the tests to verify that everything is functioning:
CMD ["conda", "run", "--name", "cheshire", "tox"]
