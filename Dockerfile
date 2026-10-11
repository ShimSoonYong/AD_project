FROM nvcr.io/nvidia/pytorch:26.09-py3

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir\
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    tqdm \
    jupyterlab \
    ipykernel

EXPOSE 8888

CMD ["/bin/bash"]
