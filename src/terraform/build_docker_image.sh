# This script builds the docker image for the label-studio application
# Automatically retrieves from the .env file:
# - POSTGRE_PASSWORD
# - POSTGRE_USER
# - POSTGRE_HOST

# TODO: Probably best to use buildkit to hide the password
source .env && \
 docker build \
 --build-arg=POSTGRE_PASSWORD=$POSTGRE_PASSWORD \
 --build-arg="POSTGRE_USER=postgres.$(terraform output -raw supabase_project_id)" \
 --build-arg="POSTGRE_HOST=aws-0-$(terraform output -raw supabase_project_region).pooler.supabase.com" \
 . \
 -t label-studio
