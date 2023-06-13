FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-client \
    libgl1 \
    git \
    build-essential \
    checkinstall \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    software-properties-common \
    curl && \
    rm -rf /apt/cache/* && apt clean

# RUN add-apt-repository -y ppa:deadsnakes/ppa && apt update && apt install -y python3.11 && \
#     curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11 && \
#     rm -rf /apt/cache/* && apt clean

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3

RUN python3 -m pip install \
    opencv-contrib-python==4.7.0.72 \
    tqdm==4.65.0 \
    tensorflow-addons==0.20.0 \
    pyyaml \
    scipy \
    tqdm \
    flake8 \
    tensorflow==2.12.*

RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
    && echo "$SNIPPET" >> "/root/.bashrc" && \
    echo "export TMPDIR=/tmp" >> "/root/.bashrc" && \
    sed -i 's/#force_color_prompt/force_color_prompt/' ~/.bashrc
