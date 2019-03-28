FROM nvcr.io/nvidia/pytorch:19.03-py3

COPY  requirements.txt /workspace

RUN pip install -r requirements.txt
