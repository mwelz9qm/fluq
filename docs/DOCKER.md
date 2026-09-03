# Running the tests with Docker

The project Docker image uses Python 3.12 and installs the exact dependency
versions recorded in `uv.lock`. Its default command runs the complete pytest
suite.

## Prerequisites

Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
When working in WSL2, enable your Linux distribution under **Docker Desktop →
Settings → Resources → WSL Integration**.

Confirm that Docker is available:

```bash
docker version
```

Run all commands below from the repository root, where `Dockerfile` is located.

## Build the image

```bash
docker build -t fluq-tests .
```

The name `fluq-tests` is a local image tag and can be replaced with another
name.

## Run all tests

```bash
docker run --rm fluq-tests
```

The container runs `pytest`, stops when testing finishes, and is deleted by
`--rm`. The `fluq-tests` image remains available for later test runs.

## Run selected tests

Run one test file:

```bash
docker run --rm fluq-tests pytest tests/test_BiasAnalyzer.py
```

Run one test function:

```bash
docker run --rm fluq-tests pytest tests/test_BiasAnalyzer.py::test_bias_analyzer_init
```

Pass other pytest options in the same way:

```bash
docker run --rm fluq-tests pytest -v
```

## Rebuild after making changes

For any changes made to the project, rebuild and run the image:

```bash
docker build -t fluq-tests .
docker run --rm fluq-tests
```

If there are dependency changes, update the lockfile locally after editing `pyproject.toml`:

```bash
uv lock
```

Then, rebuild and run the image.

Docker detects changes to `pyproject.toml` or `uv.lock` and rebuilds the
dependency layer. If a clean rebuild is needed, bypass the build cache:

```bash
docker build --no-cache -t fluq-tests .
```

## Remove the image

```bash
docker image rm fluq-tests
```

## Troubleshooting

If WSL reports that `docker` cannot be found, enable WSL integration in Docker
Desktop, select **Apply & restart**, and reopen the WSL terminal. If necessary,
run the following command from Windows PowerShell before reopening WSL:

```powershell
wsl --shutdown
```
