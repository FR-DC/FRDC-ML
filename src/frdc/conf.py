import logging
import os
from collections import OrderedDict
from pathlib import Path

import label_studio_sdk as label_studio
import requests
from dotenv import load_dotenv
from google.cloud import storage as gcs

logger = logging.getLogger(__name__)
logging.warning(
    "Initializing the project configuration, this may take a moment...\n"
    "Note that your project can be configured in the .env file in the root "
    "directory of the project."
)

# The ROOT_DIR is the root directory of the project.
# E.g. ROOT_DIR / src / frdc / conf.py is this file.
ROOT_DIR = Path(__file__).parents[2]
ENV_FILE = ROOT_DIR / ".env"

if ENV_FILE.exists():
    logger.info(f"Loading Environment Variables from {ENV_FILE.as_posix()}...")
    load_dotenv(ENV_FILE)
else:
    import shutil

    ENV_EXAMPLE_FILE = ROOT_DIR / ".env.example"
    if ENV_EXAMPLE_FILE.exists():
        shutil.copy(ENV_EXAMPLE_FILE, ENV_FILE)
        raise FileNotFoundError(
            f"Environment file not found at {ENV_FILE.as_posix()}. "
            "A new one has been created from the .env.example file.\n"
            "Set the necessary variables and re-run the script."
        )
    else:
        raise FileNotFoundError(
            f"Environment file not found at {ENV_FILE.as_posix()}. "
            "Please create one or copy the .env.example file in the GitHub "
            "repository."
        )

LOCAL_DATASET_ROOT_DIR = ROOT_DIR / "rsc"
logger.info(f"Local Dataset Save Root: {LOCAL_DATASET_ROOT_DIR.as_posix()}")

# == CONNECT TO GCS ===========================================================

os.environ["GOOGLE_CLOUD_PROJECT"] = "frmodel"
GCS_PROJECT_ID = os.environ.get("GCS_PROJECT_ID")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
logger.info(f"GCS Project: {GCS_PROJECT_ID}")
logger.info(f"GCS Bucket: {GCS_BUCKET_NAME}")
GCS_CLIENT = None
GCS_BUCKET = None

if GCS_PROJECT_ID is None or GCS_BUCKET_NAME is None:
    logger.warning("GCS_PROJECT_ID or GCS_BUCKET_NAME not set.")
else:
    try:
        logger.info("Connecting to GCS...")
        GCS_CLIENT = gcs.Client(project=GCS_PROJECT_ID)
        GCS_BUCKET = GCS_CLIENT.bucket(GCS_BUCKET_NAME)
        logger.info("Connected to GCS.")
    except Exception:
        logger.warning(
            "Couldn't connect to GCS. You will not be able to download files. "
            "Check that you've (1) Installed the GCS CLI and (2) Set up the"
            "ADC with `gcloud auth application-default login`. "
            "GCS_CLIENT will be None."
        )

# == CONNECT TO LABEL STUDIO ==================================================

LABEL_STUDIO_HOST = os.environ.get("LABEL_STUDIO_HOST")
LABEL_STUDIO_PORT = os.environ.get("LABEL_STUDIO_PORT")
LABEL_STUDIO_API_KEY = os.environ.get("LABEL_STUDIO_API_KEY", None)
LABEL_STUDIO_URL = f"http://{LABEL_STUDIO_HOST}:{LABEL_STUDIO_PORT}"
LABEL_STUDIO_CLIENT = None

logger.info(f"Label Studio URL: {LABEL_STUDIO_URL}")
logger.info("Retrieving Label Studio API Key from Environment...")

if LABEL_STUDIO_API_KEY is None or LABEL_STUDIO_API_KEY == "":
    logger.warning(
        "Env. Var. LABEL_STUDIO_API_KEY not set. "
        "You will not be able to connect to Label Studio to retrieve our "
        "datasets. \n"
        f"You can set this in your .env file @ {ENV_FILE.as_posix()}, or "
        "set it in your machine's environment variables."
    )
else:
    try:
        logger.info("Connecting to Label Studio...")
        requests.get(LABEL_STUDIO_URL)
        LABEL_STUDIO_CLIENT = label_studio.Client(
            url=LABEL_STUDIO_URL,
            api_key=LABEL_STUDIO_API_KEY,
        )
        logger.info("Connected to Label Studio.")
        try:
            logger.info("Retrieving main Label Studio Project id:1...")
            LABEL_STUDIO_CLIENT.get_project(1)
            logger.info(
                "Successfully retrieved main Label Studio Project id:1."
            )
        except requests.exceptions.HTTPError:
            logger.warning(
                "Couldn't get annotation project, "
                "live annotations won't work. "
                "Check that\n"
                "(1) Your API Key is correct.\n"
                "(2) Your API Key is for the correct LS instance.\n"
                "(3) Your .netrc is not preventing you from accessing the "
                "project. "
            )
    except requests.exceptions.ConnectionError:
        logger.warning(
            f"Could not connect to Label Studio at {LABEL_STUDIO_URL}.\n"
            f"Check that the server is running in your browser. "
            f"Label Studio features won't work. "
        )

if LABEL_STUDIO_CLIENT is None:
    logger.error(
        "Failed to connect to Label Studio, LABEL_STUDIO_CLIENT will be None."
    )

# == OTHER CONSTANTS ==========================================================

BAND_CONFIG = OrderedDict(
    {
        "WB": ("*result.tif", lambda x: x[..., 2:3]),
        "WG": ("*result.tif", lambda x: x[..., 1:2]),
        "WR": ("*result.tif", lambda x: x[..., 0:1]),
        "NB": ("result_Blue.tif", lambda x: x),
        "NG": ("result_Green.tif", lambda x: x),
        "NR": ("result_Red.tif", lambda x: x),
        "RE": ("result_RedEdge.tif", lambda x: x),
        "NIR": ("result_NIR.tif", lambda x: x),
    }
)

BAND_MAX_CONFIG: dict[str, tuple[int, int]] = {
    "WR": (0, 2**8),
    "WG": (0, 2**8),
    "WB": (0, 2**8),
    "NR": (0, 2**14),
    "NG": (0, 2**14),
    "NB": (0, 2**14),
    "RE": (0, 2**14),
    "NIR": (0, 2**14),
}
