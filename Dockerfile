FROM public.ecr.aws/docker/library/python:3.10
WORKDIR /workspace
COPY requirements.txt /workspace
RUN pip3 install --upgrade pip -r requirements.txt
COPY . /workspace
