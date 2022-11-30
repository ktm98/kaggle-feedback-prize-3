FROM kaggle/python-gpu-build

USER root


COPY requirements.txt .
RUN pip install -r requirements.txt

RUN pip uninstall -y torch 
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

RUN pip uninstall -y transformers
RUN pip install transformers==4.21.2

RUN pip uninstall -y tokenizers
RUN pip install tokenizers==0.12.1



