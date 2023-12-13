# load.gcs

> [Set-Up Google Cloud](Getting-Started.md#gcloud) to use this library.
> {style='warning'}

<tldr>
Low-level GCS utilities to automatically download and load objects from GCS.
</tldr>

## Usage

These are defined in the top-level load.gcs module.

<deflist>
<def title="list_gcs_datasets">
Lists all datasets in the bucket as a DataFrame.
This works by checking which folders have a specific file, which we call the
<code>anchor</code>.
</def>
</deflist>

> All functions below will download the file if it doesn't exist locally.
> It also checks the hash of the file, thus will not redundantly download

<deflist>
<def title="download">
Downloads a file from Google Cloud Storage and returns the local file path.
</def>
<def title="open_file">
Downloads and opens a file from Google Cloud Storage. Returns a file handle. 
</def>
<def title="open_image">
Downloads and returns the PIL image from Google Cloud Storage.
</def>
</deflist>

### Pathing

The path to specify is relative to the bucket, which is `frdc-ds` by default.

For example this filesystem on GCS:

```
# On Google Cloud Storage
frdc-ds
├── chestnut_nature_park
│   └── 20201218
│       └── 90deg
│           └── bounds.json
```

To download `bounds.json`, use `download(r"chestnut_nature_park/20201218/90deg/bounds.json")`.
By default, all files will be downloaded to `PROJ_DIR/rsc/...`.

```
# On local filesystem
PROJ_DIR
├── rsc
│   └── chestnut_nature_park
│       └── 20201218
│           └── 90deg
│               └── bounds.json
```

### Configuration {collapsible="true"}

If you need granular control over

- where the files are downloaded
- the credentials used
- the project used
- the bucket used

Then pass in a `GCSConfig` object to the utility functions.

```python
from frdc.load.gcs import download, GCSConfig

cfg = GCSConfig(credentials=...,
                local_dir=...,
                project_id=...,
                bucket_name=...,
                local_exists_ok=...)
fp = download(fp="path/to/gcs/file", config=cfg)
```

<deflist>
<def title="credentials">
<b>Google Cloud credentials.</b><br/>
A <code>google.oauth2.service_account.Credentials</code> object. See the object
documentation for more information.
</def>
<def title="local_dir">
<b>Local directory to download files to.</b><br/>
Path to a directory, or a <code>Path</code> object.
</def>
<def title="project_id">
<b>Google Cloud project ID.</b><br/>
</def>
<def title="bucket_name">
<b>Google Cloud Storage bucket name.</b><br/>
</def>
<def title="local_exists_ok">
<b>Whether to skip downloading if the file exists locally.</b><br/>
If <code>True</code>, it will skip downloading if the file exists locally.
If <code>False</code>, it will raise an error if the file exists locally.
</def>

</deflist>
