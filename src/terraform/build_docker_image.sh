# This script builds the docker image for the label-studio application
# Automatically retrieves from the .env file:
# - POSTGRE_PASSWORD
# - POSTGRE_USER
# - POSTGRE_HOST

source .env && \
 docker build \
 --build-arg=POSTGRE_PASSWORD=$POSTGRE_PASSWORD \
 --build-arg=POSTGRE_USER=$POSTGRE_USER \
 --build-arg=POSTGRE_HOST=$POSTGRE_HOST \
 . \
 -t label-studio
