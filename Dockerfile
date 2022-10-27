FROM python:3.10.4

RUN mkdir /app
WORKDIR /app
ADD requirements.txt /app/

EXPOSE 80

RUN apt-get update
RUN apt-get install libgdal-dev -y
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN apt-get -y install locales

RUN sed -i -e 's/# sv_SE.UTF-8 UTF-8/sv_SE.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL sv_SE.UTF-8
ENV LANG sv_SE.UTF-8

RUN pip install -r requirements.txt
ADD . /app/

CMD ["export", "LC_ALL='sv_SE.utf8'"]
CMD ["export", "LC_CTYPE='sv_SE.utf8'"]
CMD ["dpkg-reconfigure", "locales"]

ENTRYPOINT [ "python" ]
CMD ["application.py"]
