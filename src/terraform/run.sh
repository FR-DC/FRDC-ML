# Source and export from .env if exists
if [ -f .env ]; then
  export $(cat .env | xargs)
fi

echo "POSTGRE_USER: $(terraform output -raw pg_user)"
echo "POSTGRE_HOST: $(terraform output -raw pg_host)"

docker run -p 8080:8080 \
  -e POSTGRE_PASSWORD=$POSTGRE_PASSWORD \
  -e POSTGRE_USER="$(terraform output -raw pg_user)" \
  -e POSTGRE_HOST="$(terraform output -raw pg_host)" \
  ghcr.io/fr-dc/frdc-ml:terraform
