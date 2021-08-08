FROM python:3.8-slim-buster
WORKDIR /project
ADD . /project
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install --default-timeout=100 -r requirements.txt
EXPOSE 5000
CMD ["flask", "run", "-h", "0.0.0.0", "-p", "5000"]