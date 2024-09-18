# Stage 1: Build Stage
FROM jupyter/minimal-notebook:python-3.10 as builder

USER root

RUN apt-get update

# Install tell
RUN pip install --upgrade pip
RUN pip install tell

# Stage 2: Final Stage
FROM jupyter/minimal-notebook:python-3.10

USER root

# Copy python packages installed/built from the builder stage
COPY --from=builder /opt/conda/lib/python3.10/site-packages /opt/conda/lib/python3.10/site-packages

# To test this container locally, run:
# docker build -t tell .
# docker run --rm -p 8888:8888 tell