FROM rapidsai/rapidsai-core:21.12-cuda11.2-runtime-ubuntu20.04-py3.8

RUN apt-get update -y && apt-get install libgl1-mesa-glx nginx tmux vim -y \
    && source activate rapids \ 
    && pip install torch torchvision opencv-python transformers gensim seaborn \
    wandb efficientnet-pytorch scikit-plot scikit-image pytorch-lightning umap-learn \
    iterative-stratification pymongo python-dotenv flask flask-cors Flask-PyMongo
COPY unlabeled /etc/nginx/sites-enabled/
RUN nginx -c /etc/nginx/nginx.conf \
    && nginx -s reload \
    && mkdir workspace
WORKDIR /workspace