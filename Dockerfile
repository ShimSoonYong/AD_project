FROM nvcr.io/nvidia/pytorch:26.09-py3

WORKDIR /workspace

ENV PYTHONUNBUFFERED=1 \
  DEBIAN_FRONTEND=noninteractive

COPY requirements.txt .

RUN pip install --no-cache-dir \
  -r requirements.txt

EXPOSE 8888

CMD ["/bin/bash"]
