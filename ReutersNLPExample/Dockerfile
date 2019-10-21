FROM python:3.7

RUN pip install -U jupyter tensorflow pandas nltk scikit-learn Keras

# So any user can start the jupyter installed by root
RUN mkdir /.local && chmod -R 777 /.local

# Download example data sets
RUN python -m nltk.downloader -d /usr/local/nltk_data reuters

WORKDIR /work
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--no-browser"]
