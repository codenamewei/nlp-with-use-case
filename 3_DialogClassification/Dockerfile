FROM python:3.8.5

RUN apt-get update && apt-get upgrade -y

COPY sklearn_model.pickle /home/workdir/
COPY sklearn_vectorizer.pickle /home/workdir/
COPY skdeploy.py /home/workdir/
COPY requirements.txt /home/workdir/

WORKDIR /home/workdir/
RUN /usr/local/bin/python -m pip install --upgrade pip && pip install -r requirements.txt
CMD ["uvicorn", "skdeploy:app", "--host", "0.0.0.0", "--port", "8081"]

