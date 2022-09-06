# To build Docker images, please refer to scripts/travis/build_images.sh.
ARG BASE_IMAGE

FROM ${BASE_IMAGE} as dev
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

# Allows for log messages by `print` in Python to be immediately dumped
# to the stream instead of being buffered.
ENV PYTHONUNBUFFERED 0

COPY docker/scripts/bashrc /etc/bash.bashrc
RUN chmod a+rx /etc/bash.bashrc

RUN apt-get -qq update && apt-get -qq install -y \
        unzip \
        curl \
        git \
        software-properties-common \
        g++ \
        wget \
        build-essential \
        cmake \
        vim \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        librdmacm1 \
        libibverbs1 \
        ibverbs-providers \
        shellcheck \
        libeigen3-dev \
        clang-format > /dev/null && \
        python -m pip install --quiet --upgrade pip

# Install Go and related tools
ARG GO_MIRROR_URL=https://dl.google.com/go
ENV GOPATH /root/go
ENV PATH /usr/local/go/bin:$GOPATH/bin:$PATH
COPY docker/scripts/install_go.bash /
RUN /install-go.bash ${GO_MIRROR_URL} && rm /install-go.bash

ENTRYPOINT ["pre-commit run -a"]
