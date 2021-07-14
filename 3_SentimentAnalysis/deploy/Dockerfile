FROM python:3.8.5

RUN apt-get update && apt-get upgrade -y

COPY . /home/workdir/

WORKDIR /home/workdir/
RUN /usr/local/bin/python -m pip install --upgrade pip && pip install -r requirements.txt

EXPOSE 5000
CMD ["uvicorn", "skdeploy:app", "--host", "0.0.0.0", "--port", "5000"]

