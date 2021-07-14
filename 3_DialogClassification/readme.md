

```
docker tag sksentiment:latest codenamewei/nlp:sksentiment0.1.1
docker push codenamewei/nlp:sksentiment0.1.1

docker pull codenamewei/nlp:sksentiment0.1.1

docker build -t sksentiment .
docker run -d -p 5000:5000 codenamewei/nlp:sksentiment0.1.1  
```