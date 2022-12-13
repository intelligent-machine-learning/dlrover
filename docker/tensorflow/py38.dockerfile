FROM python:3.8.14
ARG EXTRA_PYPI_INDEX=https://pypi.org/simple

# Allows for log messages by `print` in Python to be immediately dumped
# to the stream instead of being buffered.
ENV PYTHONUNBUFFERED 0

RUN pip install tensorflow -U
RUN pip install dlrover -U
COPY model_zoo /model_zoo
