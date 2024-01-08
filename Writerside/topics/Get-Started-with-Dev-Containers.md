# Get Started with Dev Containers

Dev. Containers are a great way to get started with a project. They define all
necessary dependencies and environments, so you can just start coding within
the container.

In this article, we'll only go over **additional steps** to set up with our
project. For more information on how to use Dev Containers, please refer to
the official documentation for each IDE. Once you've set up the Dev Container,
come back here to finish the setup:

- [VSCode](https://code.visualstudio.com/docs/remote/containers).
- [IntelliJ](https://www.jetbrains.com/help/idea/connect-to-devcontainer.html)

> If you see the error `Error response from daemon: ... <!DOCTYPE`, you forgot
> the `.git` at the end of the repo URL.
{style='warning'}

## Python Environment

> Do not create a new environment
{style='warning'}

The dev environment is already created and is managed by Anaconda 
`/opt/conda/bin/conda`. 
To activate the environment, run the following command:

```bash
conda activate base
```

> Refer to your respective IDE's documentation on how to activate the
> environment.

## Mark as Sources Root (Add to PYTHONPATH)

For `import` statements to work, you need to mark the `src` folder as the
sources root. Optionally, also mark the `tests` folder as the tests root.

> Refer to your respective IDE's documentation on how to mark folders as
> sources root. (Also known as adding to the `PYTHONPATH`)

## Additional Setup

Refer to the [Getting Started](Getting-Started.md) guide for additional setup
steps such as:
- Google Cloud Application Default Credentials
- Weight & Bias API Key
- Label Studio API Key
