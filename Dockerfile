# Use Python 3.10 slim as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install FAISS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    libomp-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python deps
RUN pip install --upgrade pip
RUN pip install faiss-cpu
RUN pip install -r requirements.txt

# Expose default Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "test_tools.py", "--server.port=8501", "--server.enableCORS=false"]

