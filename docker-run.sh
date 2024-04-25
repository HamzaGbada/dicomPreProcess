#!/bin/zsh

if docker ps | grep -q dicom; then
    docker stop dicom
    echo "Stopped running 'dicom' container."
fi

if docker ps -a | grep -q dicom; then
    docker rm dicom
    echo "Removed existing 'dicom' container."
fi

if docker images -q hamzagbada18/dicom:latest; then
    docker rmi hamzagbada18/dicom:latest
    echo "Removed existing 'hamzagbada18/dicom:latest' image."
fi

docker build -t hamzagbada18/dicom:latest -f Dockerfile1 .
docker run -p 8000:8000 --name dicom hamzagbada18/dicom:latest

