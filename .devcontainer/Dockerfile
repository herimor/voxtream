FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Install additional dependencies
RUN apt update \
    && apt install -y git \
    && apt install -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy dependencies
COPY . /workspace/
RUN pip install --no-cache-dir -r requirements.txt

# Install MFA
RUN conda install -c conda-forge montreal-forced-aligner==3.3.4 -y
ENV LD_LIBRARY_PATH=/opt/conda/lib:$LD_LIBRARY_PATH
