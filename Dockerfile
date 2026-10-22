FROM nvcr.io/nvidia/pytorch:26.09-py3

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
  DEBIAN_FRONTEND=noninteractive

RUN pip install --no-cache-dir \
  -r requirements.txt \
  einops \
  scikit-learn \
  matplotlib \
  numpy \
  pandas

EXPOSE 8888

CMD ["/bin/bash"]
