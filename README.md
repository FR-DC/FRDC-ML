# FRDC-ML

**Forest Recover Digital Companion** Machine Learning Pipeline Repository

This repository contains all code regarding our models used.
This is part of the entire E2E pipeline for our product.

```mermaid
graph LR
    A[Data Collection] --> B[FRDC-ML] --> C[FRDC-UI]
```

Currently, it's heavily WIP.

## Getting Started

I highly recommend reading our [website documentation](
https://fr-dc.github.io/FRDC-ML/getting-started.html
). There contains tutorials and docs on how to use our modules.


## Dev Info

```
FRDC-ML/
    src/                    # All relevant code
        frdc/               # Package/Component Level code
            load/           # Image I/O
            preprocess/     # Image Preprocessing
            train/          # ML Training
            evaluate/       # Model Evaluation
            ...             # ...
        main.py             # Pipeline Entry Point

    tests/                  # PyTest Tests
        model-tests/        # Tests for each model
        integration-tests/  # Tests that run the entire pipeline
        unit-tests/         # Tests for each component

    poetry.lock             # Poetry managed environment file
    pyproject.toml          # Project-level information: requirements, settings, name, deployment info

    .github/                # GitHub Actions
```

## Our Architecture

This is a classic, simple Python Package architecture, however, we 
**HEAVILY EMPHASIZE** encapsulation of each stage.
That means, there should never be data that **IMPLICITLY** persists across
stages.

To illustrate this, take a look at how 
`tests/model_tests/chestnut_dec_may/train.py` is written. It pulls in relevant
modules from each stage and constructs a pipeline.


> Initially, we evaluated a few ML E2E solutions, despite them offering great
> functionality, their flexibility was
> limited. From a dev perspective, **Active Learning** was a gray area, and we
> foresee heavy shoehorning.
> Ultimately, we decided that the risk was too great, thus we resort to
> creating our own solution.

## Contributing

### Pre-commit Hooks

We use Black and Flake8 as our pre-commit hooks. To install them, run the
following commands:

```bash
poetry install
pre-commit install
```

If you're using `pip` instead of `poetry`, run the following commands:

```bash
pip install pre-commit
pre-commit install
```
