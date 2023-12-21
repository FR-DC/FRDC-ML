set -a
source .env
set +a

cml runner launch \
  --token=${GH_CML_TOKEN} \
  --labels="cml-gpu" \
  --idle-timeout="1h" --driver=github

