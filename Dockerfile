# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       latest (pip)
# ==================================================================
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    PIP_INSTALL="python -m pip --no-cache-dir install --upgrade" && \
    GIT_CLONE="git clone --depth 10" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        cmake \
        wget \
        git \
        vim \
	nano \
        libx11-dev \
        fish \
        libsparsehash-dev \
        && \
# ==================================================================
# python
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        software-properties-common \
        && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        python3.7 \
        python3.7-dev \
        && \
    wget -O ~/get-pip.py \
        https://bootstrap.pypa.io/get-pip.py && \
    python3.7 ~/get-pip.py && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python3 && \
    ln -s /usr/bin/python3.7 /usr/local/bin/python && \
    $PIP_INSTALL \
        setuptools \
        && \
    $PIP_INSTALL \
        numpy \
        scipy \
        matplotlib \
        Cython \
        && \
# ==================================================================
# pytorch
# ------------------------------------------------------------------
    $PIP_INSTALL \
        torch==1.1 -f \
        https://download.pytorch.org/whl/cu90/stable \
        && \
    $PIP_INSTALL \
        torchvision==0.3.0 \
        && \
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
    ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*



# Install python packages

WORKDIR /root


# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# nvidia runtime
COPY --from=nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04 \
 /usr/local/lib/x86_64-linux-gnu \
 /usr/local/lib/x86_64-linux-gnu

COPY --from=nvidia/opengl:1.1-glvnd-runtime-ubuntu16.04 \
 /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json \
 /usr/local/share/glvnd/egl_vendor.d/10_nvidia.json

RUN echo '/usr/local/lib/x86_64-linux-gnu' >> /etc/ld.so.conf.d/glvnd.conf && \
 ldconfig && \
 echo '/usr/local/$LIB/libGL.so.1' >> /etc/ld.so.preload && \
 echo '/usr/local/$LIB/libEGL.so.1' >> /etc/ld.so.preload

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

WORKDIR /root



#ROS
RUN git clone https://github.com/keremyldrr/3D-Instance-Segmentation-with-LDLS-YOLACT.git
WORKDIR 3D-Instance-Segmentation-with-LDLS-YOLACT
RUN apt update
#RUN apt-get install -y  python3-pip
RUN git checkout dev

RUN pip install opencv-python
RUN apt update && apt install -y libsm6 libxext6
RUN apt-get install -y libxrender-dev
RUN add-apt-repository ppa:graphics-drivers
RUN pip install cupy-cuda101
RUN pip install -r requirements.txt
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.1.105-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu1804_10.1.105-1_amd64.deb

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y cuda-toolkit-10-1
RUN echo "# set PATH for cuda 10.1 installation \
if [ -d "/usr/local/cuda-10.1/bin/" ]; then \
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}} \
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} \
fi" >> .bashrc \


RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.2/lib64/

CMD jupyter-notebook --no-browser --ip=0.0.0.0 --port=8080  --allow-root





