FROM python:3.5

RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        libatlas-base-dev gfortran nginx supervisor

RUN apt-get update \
&& apt-get -y install libxrender1 \
&& apt-get -y install libsm6 libxext6 \
&& mkdir -p /project 

RUN pip3 install uwsgi
RUN mkdir -p /project
COPY ./requirements.txt /project/requirements.txt

RUN pip3 install -r /project/requirements.txt

RUN useradd --no-create-home nginx

RUN rm /etc/nginx/sites-enabled/default
RUN rm -r /root/.cache

COPY nginx.conf /etc/nginx/
COPY cors.conf /etc/nginx/
COPY flask-site-nginx.conf /etc/nginx/conf.d/
COPY uwsgi.ini /etc/uwsgi/
COPY supervisord.conf /etc/


COPY mydata.txt /project/
COPY coco.names /project/
COPY yolov3.weights /project/
COPY yolov3.cfg /project/
COPY wsgi.py /project/
COPY __init__.py /project/

WORKDIR /project

CMD ["/usr/bin/supervisord"]
