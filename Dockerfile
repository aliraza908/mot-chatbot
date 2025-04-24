# Use official Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    swig \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run Streamlit app
CMD ["streamlit", "run", "test_tools.py", "--server.port=8080", "--server.address=0.0.0.0"]
