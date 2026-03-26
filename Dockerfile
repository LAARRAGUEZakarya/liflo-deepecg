FROM runpod/base:0.4.0-cuda11.8.0

# System deps
RUN apt-get update && apt-get install -y git wget curl && rm -rf /var/lib/apt/lists/*

# Miniconda with Python 3.10 (required by DeepECG)
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

# Clone DeepECG and fairseq-signals
RUN git clone https://github.com/HeartWise-AI/DeepECG_Docker /workspace/DeepECG_Docker
RUN git clone https://github.com/HeartWise-AI/fairseq-signals /workspace/DeepECG_Docker/fairseq-signals

WORKDIR /workspace/DeepECG_Docker

# Install dependencies in correct order
RUN pip install --upgrade pip && \
    pip install "omegaconf==2.1.1" "hydra-core==1.1.2" && \
    pip install "tokenizers==0.13.3" --prefer-binary && \
    pip install "transformers==4.30.0" && \
    pip install torch==2.5.1+cu118 --index-url https://download.pytorch.org/whl/cu118 && \
    pip install wfdb pydicom Pillow pandas numpy statsmodels scipy scikit-learn && \
    pip install runpod

# Environment
ENV PYTHONPATH=/workspace/DeepECG_Docker:/workspace/DeepECG_Docker/fairseq-signals
ENV MPLBACKEND=Agg

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
