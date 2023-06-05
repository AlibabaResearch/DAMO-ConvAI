ARG BASE_IMAGE

# ------------------------
# Target: dev
# ------------------------
FROM $BASE_IMAGE as dev

ARG TOOLKIT_USER_ID=13011
ARG TOOLKIT_GROUP_ID=13011

RUN apt-get update \
    # Required to save git hashes
    && apt-get install -y -q git curl unzip make gettext \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV XDG_DATA_HOME=/app/.local/share \
    XDG_CACHE_HOME=/app/.cache \
    XDG_BIN_HOME=/app/.local/bin \
    XDG_CONFIG_HOME=/app/.config
RUN mkdir -p $XDG_DATA_HOME \
    && mkdir -p $XDG_CACHE_HOME \
    && mkdir -p $XDG_BIN_HOME \
    && mkdir -p $XDG_CONFIG_HOME \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app

# Install C++ toolchain, Facebook thrift, and dependencies
RUN curl https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-get update \
    && apt-get install -y --no-install-recommends software-properties-common \
    && apt-add-repository "deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-9 main" \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        binfmt-support libllvm9 llvm-9 llvm-9-dev llvm-9-runtime llvm-9-tools python-chardet python-pygments python-yaml \
        g++ \
        cmake \
        libboost-all-dev \
        libevent-dev \
        libdouble-conversion-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libiberty-dev \
        liblz4-dev \
        liblzma-dev \
        libsnappy-dev \
        make \
        zlib1g-dev \
        binutils-dev \
        libjemalloc-dev \
        libssl-dev \
        pkg-config \
        libunwind-dev \
        libmysqlclient-dev \
        bison \
        flex \
        libsodium-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/zstd /app/third_party/zstd/
RUN cd /app/third_party/zstd \
    && make -j4 \
    && make install \
    && make clean
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/fmt /app/third_party/fmt/
RUN cd /app/third_party/fmt/ \
    && mkdir _build \
    && cd _build \
    && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=off -DBUILD_TESTS=off ../. \
    && make -j4 \
    && make install \
    && cd .. \
    && rm -rf _build
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/folly /app/third_party/folly/
RUN pip install cython \
    && cd /app/third_party/folly \
    && mkdir _build \
    && cd _build \
    && cmake -DBUILD_SHARED_LIBS=ON -DPYTHON_EXTENSIONS=ON -DBUILD_EXAMPLES=off -DBUILD_TESTS=off ../. \
    && make -j4 \
    && make install \
    && cp folly/cybld/dist/folly-0.0.1-cp37-cp37m-linux_x86_64.whl /app/ \
    && chown $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/folly-0.0.1-cp37-cp37m-linux_x86_64.whl \
    && pip install /app/folly-0.0.1-cp37-cp37m-linux_x86_64.whl \
    && cd .. \
    && rm -rf _build
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/rsocket-cpp /app/third_party/rsocket-cpp/
RUN cd /app/third_party/rsocket-cpp \
    && mkdir _build \
    && cd _build \
    && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=off -DBUILD_TESTS=off ../. \
    && make -j4 \
    && make install \
    && cd .. \
    && rm -rf _build
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/fizz /app/third_party/fizz/
RUN cd /app/third_party/fizz \
    && mkdir _build \
    && cd _build \
    && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=off -DBUILD_TESTS=off ../fizz \
    && make -j4 \
    && make install \
    && cd .. \
    && rm -rf _build
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/wangle /app/third_party/wangle/
RUN cd /app/third_party/wangle \
    && mkdir _build \
    && cd _build \
    && cmake -DBUILD_SHARED_LIBS=ON -DBUILD_EXAMPLES=off -DBUILD_TESTS=off ../wangle \
    && make -j4 \
    && make install \
    && cd .. \
    && rm -rf _build
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/fbthrift /app/third_party/fbthrift/
RUN cd /app/third_party/fbthrift \
    && mkdir _build \
    && cd _build \
    && cmake \
        -DBUILD_SHARED_LIBS=ON \
        -DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
        -DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") \
        -Dthriftpy3=ON \
        ../. \
    && make -j4 \
    && DESTDIR=/ make install \
    && cp thrift/lib/py3/cybld/dist/thrift-0.0.1-cp37-cp37m-linux_x86_64.whl /app/ \
    && chown $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/thrift-0.0.1-cp37-cp37m-linux_x86_64.whl \
    && pip install /app/thrift-0.0.1-cp37-cp37m-linux_x86_64.whl \
    && cd .. \
    && rm -rf _build

# Install Rust toolchain
ENV RUSTUP_HOME=/app/.local/rustup \
    CARGO_HOME=/app/.local/cargo \
    PATH=/app/.local/cargo/bin:$PATH
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        gcc \
        libc6-dev \
        wget \
        ; \
    \
    url="https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init"; \
    wget "$url"; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --default-toolchain nightly-2021-06-01; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version; \
    rm -rf /var/lib/apt/lists/*; \
    chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.local/cargo; \
    chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.local/rustup;

# Install Haskell toolchain
ENV BOOTSTRAP_HASKELL_NONINTERACTIVE=yes \
    BOOTSTRAP_HASKELL_NO_UPGRADE=yes \
    GHCUP_USE_XDG_DIRS=yes \
    GHCUP_INSTALL_BASE_PREFIX=/app \
    CABAL_DIR=/app/.cabal \
    PATH=/app/.cabal/bin:/app/.local/bin:$PATH
RUN buildDeps=" \
        curl \
        "; \
    deps=" \
        libtinfo-dev \
        libgmp3-dev \
        "; \
    apt-get update \
    && apt-get install -y --no-install-recommends $buildDeps $deps \
    && curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh \
    && ghcup install ghc \
    && ghcup install cabal \
    && cabal update \
    && apt-get install -y --no-install-recommends git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && git clone https://github.com/haskell/cabal.git \
    && cd cabal \
    && git checkout f5f8d933db229d30e6fc558f5335f0a4e85d7d44 \
    && sed -i 's/3.5.0.0/3.6.0.0/' */*.cabal \
    && cabal install cabal-install/ \
        --allow-newer=Cabal-QuickCheck:Cabal \
        --allow-newer=Cabal-described:Cabal \
        --allow-newer=Cabal-tree-diff:Cabal \
        --allow-newer=cabal-install:Cabal \
        --allow-newer=cabal-install-solver:Cabal \
    && cd .. \
    && rm -rf cabal/ \
    && rm -rf /app/.cabal/packages/* \
    && rm -rf /app/.cabal/logs/* \
    && rm -rf /app/.cache/ghcup \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.cabal \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.local/bin \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.local/share/ghcup

# Build Facebook hsthrift
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/hsthrift /app/third_party/hsthrift/
RUN cd /app/third_party/hsthrift \
    && make thrift-cpp \
    && cabal update \
    && cabal build exe:thrift-compiler \
    && make thrift-hs \
    && cabal install exe:thrift-compiler \
    && cabal clean \
    && rm -rf /app/.cabal/packages/* \
    && rm -rf /app/.cabal/logs/* \
    && chown -h $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.cabal/bin/thrift-compiler \
    && find /app/.cabal/store/ghc-8.10.*/ -maxdepth 2 -type d -group root -exec chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID {} \; \
    && find . -group root -exec chown $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID {} \;

# Install misc utilities and add toolkit user
ENV LANG=en_US.UTF-8
RUN apt update && \
    apt install -y \
        zsh fish gnupg lsb-release \
        ca-certificates supervisor openssh-server bash ssh tmux jq \
        curl wget vim procps htop locales nano man net-tools iputils-ping \
        openssl libicu[0-9][0-9] libkrb5-3 zlib1g gnome-keyring libsecret-1-0 desktop-file-utils x11-utils && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg && \
    echo \
        "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \
        $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update && \
    apt-get install -y docker-ce docker-ce-cli containerd.io && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && \
    locale-gen && \
    useradd -m -u $TOOLKIT_USER_ID -s /bin/bash --non-unique toolkit && \
    passwd -d toolkit && \
    useradd -m -u $TOOLKIT_USER_ID -s /bin/bash --non-unique console && \
    passwd -d console && \
    useradd -m -u $TOOLKIT_USER_ID -s /bin/bash --non-unique _toolchain && \
    passwd -d _toolchain && \
    useradd -m -u $TOOLKIT_USER_ID -s /bin/bash --non-unique coder && \
    passwd -d coder && \
    chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /run /etc/shadow /etc/profile && \
    apt autoremove --purge && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    echo ssh >> /etc/securetty && \
    rm -f /etc/legal /etc/motd

# Build Huggingface tokenizers Rust libraries
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID third_party/tokenizers /app/third_party/tokenizers/
RUN cd /app/third_party/tokenizers \
    && rustup --version \
    && cargo --version \
    && rustc --version \
    && cargo build --release \
    && cp target/release/libtokenizers_haskell.so /usr/lib/ \
    && rm -rf target \
    && find /app/.local/cargo -group root -exec chown $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID {} \;
ENV TOKENIZERS_PARALLELISM=false

# Install Python toolchain
ENV PYTHONPATH=/app
RUN pip install --no-cache-dir pre-commit "poetry==1.1.7"
# Disable virtualenv creation to install our dependencies system-wide.
RUN poetry config virtualenvs.create false
# Config file is not readable by other users by default, which prevents
# it from being read on Drone, therefore make it readable.
RUN chmod go+r $XDG_CONFIG_HOME/pypoetry/config.toml
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID pyproject.toml poetry.lock /app/
RUN poetry install --extras "deepspeed" \
    && pip install /app/folly-0.0.1-cp37-cp37m-linux_x86_64.whl \
    && pip install /app/thrift-0.0.1-cp37-cp37m-linux_x86_64.whl \
    && rm -rf $XDG_CACHE_HOME/pip \
    && rm -rf $XDG_CACHE_HOME/pypoetry \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID $XDG_CONFIG_HOME/pypoetry

# Unfortunately, nltk doesn't look in XDG_DATA_HOME, so therefore /usr/local/share
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords

# ------------------------
# Target: train
# ------------------------
FROM dev as train

ARG TOOLKIT_USER_ID=13011
ARG TOOLKIT_GROUP_ID=13011

# Misc environment variables
ENV HF_HOME=/transformers_cache

# Copy Seq-to-seq code
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./seq2seq /app/seq2seq/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./tests /app/tests/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./third_party/spider /app/third_party/spider/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./third_party/test_suite /app/third_party/test_suite/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./configs /app/configs/

# ------------------------
# Target: eval
# ------------------------
FROM dev as eval

ARG TOOLKIT_USER_ID=13011
ARG TOOLKIT_GROUP_ID=13011

# Add thrift file
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID picard.thrift /app/

# Build Cython code
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID gen-cpp2 /app/gen-cpp2/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID gen-py3 /app/gen-py3/
RUN thrift1 --gen mstch_cpp2 picard.thrift \
    && thrift1 --gen mstch_py3 picard.thrift \
    && cd gen-py3 && python setup.py build_ext --inplace \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/gen-py3 /app/gen-cpp2
ENV PYTHONPATH=$PYTHONPATH:/app/gen-py3 \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/app/gen-py3/picard

# Build and install Picard
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID cabal.project fb-util-cabal.patch /app/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID gen-hs /app/gen-hs/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID picard /app/picard/
RUN cabal update \
    && cd third_party/hsthrift \
    && make THRIFT_COMPILE=thrift-compiler thrift-cpp thrift-hs \
    && cd ../.. \
    && thrift-compiler --hs --use-hash-map --use-hash-set --gen-prefix gen-hs -o . picard.thrift \
    && patch -p 1 -d third_party/hsthrift < ./fb-util-cabal.patch \
    && cabal install --overwrite-policy=always --install-method=copy exe:picard \
    && chown $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/.cabal/bin/picard \
    && cabal clean \
    && rm -rf /app/third_party/hsthrift/compiler/tests \
    && rm -rf /app/.cabal/packages/* \
    && rm -rf /app/.cabal/logs/* \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/picard/ \
    && chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID /app/gen-hs/ \
    && find /app/.cabal/store/ghc-8.10.*/ -maxdepth 2 -type d -group root -exec chown -R $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID {} \; \
    && find /app/.cabal/store/ghc-8.10.*/ -maxdepth 2 -type f -group root -exec chown $TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID {} \;

# Misc environment variables
ENV HF_HOME=/transformers_cache

# Copy Seq-to-seq code
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./seq2seq /app/seq2seq/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./tests /app/tests/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./third_party/spider /app/third_party/spider/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./third_party/test_suite /app/third_party/test_suite/
COPY --chown=$TOOLKIT_USER_ID:$TOOLKIT_GROUP_ID ./configs /app/configs/

# Test Picard
RUN python /app/tests/test_picard_client.py \
    && rm -rf /app/seq2seq/__pycache__ \
    && rm -rf /app/gen-py3/picard/__pycache__
