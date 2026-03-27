FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install "omegaconf==2.1.1" "hydra-core==1.1.2" && \
    pip install "tokenizers==0.13.3" --prefer-binary && \
    pip install "transformers==4.30.0" && \
    pip install wfdb pydicom Pillow pandas numpy statsmodels scipy scikit-learn runpod

RUN git clone https://github.com/HeartWise-AI/DeepECG_Docker /workspace/DeepECG_Docker && \
    git clone https://github.com/HeartWise-AI/fairseq-signals /workspace/DeepECG_Docker/fairseq-signals && \
    ln -s /workspace/DeepECG_Docker/fairseq-signals /fairseq-signals

ENV PYTHONPATH=/workspace/DeepECG_Docker:/workspace/DeepECG_Docker/fairseq-signals
ENV MPLBACKEND=Agg

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
