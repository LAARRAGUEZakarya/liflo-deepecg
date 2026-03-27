FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev git wget && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "omegaconf==2.1.1" "hydra-core==1.1.2" && \
    pip install "tokenizers==0.13.3" --prefer-binary && \
    pip install "transformers==4.30.0" && \
    pip install torch==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install wfdb pydicom Pillow pandas numpy statsmodels scipy scikit-learn runpod

RUN git clone https://github.com/HeartWise-AI/DeepECG_Docker /workspace/DeepECG_Docker && \
    git clone https://github.com/HeartWise-AI/fairseq-signals /workspace/DeepECG_Docker/fairseq-signals

ENV PYTHONPATH=/workspace/DeepECG_Docker:/workspace/DeepECG_Docker/fairseq-signals
ENV MPLBACKEND=Agg

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
