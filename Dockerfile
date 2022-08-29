FROM nvcr.io/nvidia/pytorch:22.08-py3

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

RUN pip install pyvirtualdisplay tqdm

# Install my-extra-package-1 and my-extra-package-2
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb \
        && \
        rm -rf /var/lib/apt/lists/
