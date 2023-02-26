FROM nvcr.io/nvdlfwea/jax/jax:23.01-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        build-essential \
        curl \
        git \
        libomp-dev \
        ffmpeg \
        libjpeg-turbo8 \
        libblas-dev \
        liblapack-dev \
        gfortran \
        libblosc-dev \
        libbrotli-dev \ 
        bzip2 \
        libc-ares-dev \
        libcfitsio-dev \
        dbus \
        expat \
        fontconfig \
        fonttools \
        freetype \
        libicu-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstreamer-plugins-bad1.0-dev gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-libav gstreamer1.0-doc gstreamer1.0-tools gstreamer1.0-x gstreamer1.0-alsa gstreamer1.0-gl gstreamer1.0-gtk3 gstreamer1.0-qt5 gstreamer1.0-pulseaudio \
        giflib-tools \
        libjpeg-dev \
        liblcms2-dev \
        liblerc-dev \
        libaec-dev \
        libblas-dev \
        libcblas-dev \
        libcurl-dev \
        libdeflate-tools \
        libedit-dev \
        libev-dev \
        libffi-dev \
        libgcc-7-dev \
        libgfortran4 \
        libnghttp2-dev \
        libpng-dev \
        zlib1g-dev \
        libssh2-1-dev \
        libstdc++6 \
        libtiff-dev \
        uuid-dev \
        libwebp-dev \
        libxcb-randr0-dev libxcb-xtest0-dev libxcb-xinerama0-dev libxcb-shape0-dev libxcb-xkb-dev \
        libxml2-dev \
        libzopfli-dev \
        lz4 \
        intel-mkl \
        libncurses5-dev libncursesw5-dev \
        openjpeg-tools \
        libssl-dev \ 
        qt5-default \
        libreadline-dev \
        sqlite3 \
        python3-tk \
        xorg \
        openbox \
        xz-utils \
        python-yaml \
        zlib1g-dev \
        zstd


















#brunsli , charls igl jxrlib ld_impl_linux-64   - libgfortran-ng=7.5.0 libwebp-base llvm-openmp   - pcre=8.45




# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN wget https://developer.download.nvidia.com/compute/cuda/12.0.1/local_installers/cuda-repo-ubuntu2004-12-0-local_12.0.1-525.85.12-1_amd64.deb
# RUN dpkg -i cuda-repo-ubuntu2004-12-0-local_12.0.1-525.85.12-1_amd64.deb
# RUN cp /var/cuda-repo-ubuntu2004-12-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
# RUN apt-get -y update
# RUN apt-get -y install cuda



RUN python3 -m  pip  install torch torchvision torchaudio torch-geometric --extra-index-url https://download.pytorch.org/whl/cpu

# # RUN python3 -m pip install jaxlib --upgrade
# # RUN python3 -m pip install jax --upgrade

RUN python3 -m  pip install \
    numba==0.56.4+1.g772622d0d\
    # cupy \
    black==23.1.0\
    pandas==1.5.2 \
    numpy==1.22.2 \
    comet-ml==3.32.1 \
    h5py==3.6.0 \
    SimpleITK==2.2.1\
    batchgenerators==0.24 \
    dask==2022.11.1 \
    nibabel==5.0.0\
    # intensity-normalization[ants] \
    # warp-lang \
    report_guided_annotation==0.2.7\
    # ott-jax \ 
    # evojax[extra] \
    # jraph \
    flax==0.6.3 \
    toolz==0.12.0\
    ipyparallel==8.4.1\
    optuna==3.1.0\
    # equinox \
    einops==0.6.0\
    torchio==0.18.87 \
    jax-triton==0.1.2 \
    opencv-python==4.7.0.68 \
    dm-pix==0.3.4 \
    visu3d==1.4.0 \
    ml-collections==0.1.1 \
    #https://stackoverflow.com/questions/63938552/how-to-run-tensorboard-in-vscode
    tensorboard-plugin-profile \
    jax-smi==1.0.2 \
    tensorly==0.8.0 \
    tqdm==4.64.1 \
    kymatio==0.3.0 \
    jaxwt \
    matplotlib


RUN python3 -m  pip install \
    absl-py==1.0.0 \
    cloudpickle==2.0.0 \
    cycler==0.11.0 \
    cytoolz==0.11.0 \
    dask-core==2021.10.0 \
    fsspec==2022.1.0 \
    imagecodecs==2021.8.26 \
    imageio==2.9.0 \
    intel-openmp==2021.4.0 \
    kiwisolver==1.3.1 \
    krb5==1.19.2 \
    locket==0.2.1 \
    mkl-service==2.4.0 \
    mkl_fft==1.3.1 \
    mkl_random==1.2.2 \
    munkres==1.1.4 \
    networkx==2.6.3 \
    olefile==0.46 \
    opt_einsum==3.3.0 \
    packaging==21.3 \
    partd==1.2.0 \
    pillow==8.4.0 \
    polyscope==1.3.0 \
    pyparsing==3.0.4 \
    pyqt==5.9.2 \
    dateutil \
    flatbuffers \
    abi \
    pywavelets==1.1.1 \
    pyyaml==6.0 \
    scikit-image==0.18.3 \
    scipy==1.7.3 \
    setuptools==58.0.4 \
    sip==4.19.13 \
    six==1.16.0 \
    snappy==1.1.8 \
    tifffile==2021.7.2 \
    tornado==6.1 \
    typing_extensions==4.0.1 \
    tzdata==2021e \
    wheel==0.37.1 \
    tornado==6.1














RUN git clone https://github.com/pyro-ppl/numpyro.git ${HOME}/externalRepos/numPyro && \
    cd ${HOME}/externalRepos/numPyro && \
    pip install -e .[dev] 
# install jax/jaxlib first for CUDA support


# RUN python3 -m  pip install cython-pocketfft

#RUN python3 -m  pip install numpyro[cuda]==0.11.0 -f https://storage.googleapis.com/jax-releases/Jax_cuda_med_releases.html



# RUN python3 -m  pip install monai-weekly


# RUN python3 -m pip uninstall -y horovod
# RUN HOROVOD_WITH_PYTORCH=1 python3 -m  pip install --no-cache-dir horovod[pytorch]

RUN git clone https://github.com/SuperElastix/SimpleElastix ${HOME}/externalRepos/elastix

# RUN ["chmod", "+x", "/home/sliceruser/externalRepos/elastix"]
#from https://installati.one/ubuntu/22.04/libpangoxft-1.0-0/
RUN apt update
RUN apt -y install libtbb-dev
#from https://askubuntu.com/questions/409310/error-while-loading-shared-libraries-cannot-open-shared-object-file-no-such-fi
RUN apt -y install libgtk-3-dev
RUN apt update

RUN apt-get -y install libidn11-dev  \
                libglu1-mesa-dev  \
                libpangox-1.0-dev  \
                libpangoxft-1.0-0

RUN mkdir ${HOME}/elastixBase && \
cd ${HOME}/elastixBase && \
wget https://github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2&& \
tar -xf elastix-5.0.1-linux.tar.bz2

RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/bin/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 ${HOME}/elastixBase/elastix-5.0.1-linux/libANNlib-5.0.so.1
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so /usr/local/lib/libANNlib-5.0.so
RUN cp ${HOME}/elastixBase/elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 /usr/local/lib/libANNlib-5.0.so.1

RUN ldconfig
# RUN apt-get install -q -y vim nano 

#install go for jax memory profiling


ENV GO_VERSION=1.19.4

RUN apt-get update
RUN apt-get install -y wget git gcc

RUN wget -P /tmp "https://dl.google.com/go/go${GO_VERSION}.linux-amd64.tar.gz"

RUN tar -C /usr/local -xzf "/tmp/go${GO_VERSION}.linux-amd64.tar.gz"
RUN rm "/tmp/go${GO_VERSION}.linux-amd64.tar.gz"

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"

ENV GOPATH /go
ENV PATH $GOPATH/bin:/usr/local/go/bin:$PATH
RUN mkdir -p "$GOPATH/src" "$GOPATH/bin" && chmod -R 777 "$GOPATH"


RUN apt install graphviz
RUN go install github.com/google/pprof@latest
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.87
RUN apt install nautilus -y
#ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
#/usr/local/cuda/lib64
#/usr/local/cuda/extras/CUPTI

# ENV LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/NsightSystems-cli-2022.5.1/target-linux-x64:$LD_LIBRARY_PATH

#/usr/local/cuda-12.0/lib64/

# find ~/ -type f -name "libcupti.so"

# find . -type f -name "libcupti.so"

# ENV cudnn_version=8.6.0.163
# ENV cuda_version=cuda11.6
# libcupti.so.11.8
# /usr/local/cuda/extras/CUPTI/lib64

# RUN apt-get install libcudnn8_8.6.0.163-1+cuda11.6
# RUN apt-get install libcudnn8-dev_8.6.0.163-1+cuda11.6
# RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
# RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

# # install python3-pip
# RUN apt update && apt install python3-pip -y

# RUN pip install "jax[cuda]" -f https://storage.googleapis.com/jax-releases/Jax_cuda_med_releases.html

#FROM nvdlfwea/jax/jax

# RUN apt update && apt install python3-pip -y
# RUN git clone --recursive https://github.com/uber/horovod.git
# RUN cd horovod \
#     python setup.py clean \
#     python setup.py bdist_wheel
# ENV HOROVOD_WITH_PYTORCH=1

# RUN python3 -m  pip install horovod==0.26.1
# RUN apt install -y build-essential
# RUN apt-get update
# RUN apt install -y lsb-release wget software-properties-common gnupg
# RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"