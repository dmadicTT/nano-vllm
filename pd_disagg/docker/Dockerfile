FROM ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    git \
    cmake \
    build-essential \
    ninja-build \
    libssl-dev \
    libasio-dev \
    linux-modules-extra-$(uname -r) \
    rdma-core \
    ibverbs-utils \
    perftest \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch \
    transformers \
    numpy \
    accelerate \
    huggingface-hub

# Build Mooncake
WORKDIR /opt
RUN git clone https://github.com/kvcache-ai/Mooncake.git && \
    cd Mooncake && \
    mkdir build && cd build && \
    cmake .. -DUSE_CUDA=OFF -DBUILD_TEST=OFF && \
    make -j$(nproc)

# Set library paths
ENV MOONCAKE_BUILD=/opt/Mooncake/build
ENV LD_LIBRARY_PATH="${MOONCAKE_BUILD}/mooncake-transfer-engine/src:${MOONCAKE_BUILD}/mooncake-common:${LD_LIBRARY_PATH}"
ENV PYTHONPATH="${MOONCAKE_BUILD}/mooncake-integration:${PYTHONPATH}"

# Copy PD disaggregation code
COPY src/ /opt/pd_disagg/
WORKDIR /opt/pd_disagg

# Expose ports
EXPOSE 8050 8060

# Entry point script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
