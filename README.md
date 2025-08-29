# tis_fastmri

This project uses `uv` for package management.

## Installation

To install `uv`, run the following command:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you are using windows or you have trouble to install please refer to [uv's installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Usage

To install the dependencies for this project, run the following command in this package's root directory:

```bash
uv sync
```

To activate the virtual environment, run in this package's root directory:

```bash
source .venv/bin/activate
```
