# Use a larger base image with build tools for the first stage
FROM python:3.9-slim-buster AS builder

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    curl

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Final stage: Use a smaller base image for the final build
FROM python:3.9-slim-buster

# Set the working directory
WORKDIR /app

# Copy the dependencies and application files from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Expose port 80
EXPOSE 80

# Set an environment variable to indicate the code is running inside Docker
ENV IN_DOCKER=1

# Run the application
CMD ["sh", "-c", "python setup.py && python -m app"]
