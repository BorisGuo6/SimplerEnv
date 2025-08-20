FROM qudelin/simpler-env

# Configure environment variables
ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL="egl"
# ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies and set up Python in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libglib2.0-0 libgl1-mesa-glx libegl1-mesa ffmpeg
