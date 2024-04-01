FROM ubuntu:22.04
WORKDIR /app

RUN apt-get update && apt-get install -y python3-pip

RUN pip3 install label-studio

# NOTE
# This doesn't automatically port forward the port
# to the host machine. You need to do that manually
EXPOSE 8080

ENV DJANGO_DB=default
ENV POSTGRE_NAME=postgres
ENV POSTGRE_PORT=5432

ENTRYPOINT ["label-studio"]

# This is deprecated code that is no longer needed
# This is used if you are building a static image with predefined
# Postgres credentials, however it's best if we run the image with environment
# variables passed in from the host machine
#
# E.g.
# docker run image_tag \
#   -e POSTGRE_PASSWORD=pass
#   -e POSTGRE_USER=user
#   -e POSTGRE_HOST=host
#ARG POSTGRE_PASSWORD
#ARG POSTGRE_USER
#ARG POSTGRE_HOST
#
## Assert that the POSTGRE_PASSWORD and POSTGRE_USER
#RUN test -n "$POSTGRE_PASSWORD" || (echo "POSTGRE_PASSWORD is not set" && exit 1)
#RUN test -n "$POSTGRE_USER" || (echo "POSTGRE_USER is not set" && exit 1)
#RUN test -n "$POSTGRE_HOST" || (echo "POSTGRE_HOST is not set" && exit 1)
#ENV POSTGRE_PASSWORD=$POSTGRE_PASSWORD
#ENV POSTGRE_USER=$POSTGRE_USER
#ENV POSTGRE_HOST=$POSTGRE_HOST
