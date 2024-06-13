import os
import time
from pathlib import Path

import label_studio_sdk

THIS_DIR = Path(__file__).parent

# This is your API token. I put mine here, which is OK only if you're in a
# development environment. Otherwise, do not.
dev_api_key = os.getenv("REPLICA_LABEL_STUDIO_API_KEY")
prd_api_key = os.getenv("LABEL_STUDIO_API_KEY")
dev_url = "http://localhost:8082"
prd_url = "http://localhost:8080"

# We can initialize the sdk using this following.
# The client is like the middleman between you as a programmer, and the
# Label Studio (LS) server.
dev_client = label_studio_sdk.Client(url=dev_url, api_key=dev_api_key)
prd_client = label_studio_sdk.Client(url=prd_url, api_key=prd_api_key)

# This is the labelling interface configuration.
# We can save it somewhere as an XML file then import it too
dev_config = (THIS_DIR / "default_config.xml").read_text()

# %%
print("Creating Development Project...")
# Creates the project, note to set the config here
dev_proj = dev_client.create_project(
    title="FRDC Replica",
    description="This is the replica project of FRDC. It's ok to break this.",
    label_config=dev_config,
    color="#FF0025",
)
# %%
print("Adding Import Source...")
# This links to our GCS as an import source
dev_storage = dev_proj.connect_google_import_storage(
    bucket="frdc-ds",
    regex_filter=".*.jpg",
    google_application_credentials=(
        THIS_DIR / "frmodel-943e4feae446.json"
    ).read_text(),
    presign=False,
    title="Source",
)
time.sleep(5)
# %%
print("Syncing Storage...")
# Then, we sync it so that all the images appear as annotation targets
dev_proj.sync_storage(
    storage_type=dev_storage["type"],
    storage_id=dev_storage["id"],
)
time.sleep(5)
# %%
print("Retrieving Tasks...")
prd_proj = prd_client.get_project(id=1)
prd_tasks = prd_proj.get_tasks()
dev_tasks = dev_proj.get_tasks()
# %%
# This step copies over the annotations from the production to the development
# This creates it as a "prediction"
print("Copying Annotations...")
for prd_task in prd_tasks:
    # For each prod task, we find the corresponding (image) file name
    prd_fn = prd_task["storage_filename"]

    # Then, we find the corresponding task in the development project
    dev_tasks_matched = [
        t for t in dev_tasks if t["storage_filename"] == prd_fn
    ]

    # Do some error handling
    if len(dev_tasks_matched) == 0:
        print(f"File not found in dev: {prd_fn}")
        continue
    if len(dev_tasks_matched) > 1:
        print(f"Too many matches found in dev: {prd_fn}")
        continue

    # Get the first match
    dev_task = dev_tasks_matched[0]

    # Only get annotations by evening
    prd_ann = [
        ann
        for ann in prd_task["annotations"]
        if "dev_evening" in ann["created_username"]
    ][0]

    # Create the prediction using the result from production
    dev_proj.create_prediction(
        task_id=dev_task["id"],
        result=prd_ann["result"],
        model_version="API Testing Prediction",
    )

print("Done!")
