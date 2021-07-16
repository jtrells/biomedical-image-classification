FROM rapidsai/rapidsai:21.06-cuda11.2-runtime-ubuntu20.04-py3.8

RUN apt-get update -y && apt-get install libgl1-mesa-glx -y
RUN pip install torch torchvision opencv-python transformers gensim seaborn wandb efficientnet-pytorch scikit-plot scikit-image pytorch-lightning umap-learn iterative-stratification fastparquet
# fastparquet failing
RUN mkdir workspace
WORKDIR /workspace