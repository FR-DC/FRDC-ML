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
