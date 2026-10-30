# Contributing to DocuQuery

## Set up the project

1. Get the source code:

   ```bash
   git clone https://github.com/nnamanikester/docuquery.git docuquery
   cd docuquery
   ```

2. Install the dependencies:

   ```bash
   uv sync
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate        # macOS and Linux
   .venv\Scripts\activate           # Windows
   ```

4. Run DocuQuery:

   ```bash
   docuquery --help
   ```

## Before you commit

Run the tests, then check and format the code:

```bash
pytest
ruff check --fix
ruff format
```

## Run the database tests

The database tests run only when `DOCUQUERY_TEST_POSTGRES_URL` is set. To run them, start a separate test database:

```bash
docker run -d --rm --name docuquery-test -e POSTGRES_PASSWORD=test \
  -p 55432:5432 pgvector/pgvector:pg17
DOCUQUERY_TEST_POSTGRES_URL=postgresql://postgres:test@localhost:55432/postgres pytest
```

> **CAUTION:** The database tests delete all DocuQuery tables in the test database. Do not use a database that contains data that you want to keep.

## Releasing

DocuQuery is released automatically when a pull request is merged into `main`. Do not upload to PyPI by hand, and do not push version tags by hand.

### Choose a release label

Each pull request must have exactly one release label. A new pull request gets `release: patch`. Change the label if necessary.

| Label | Use it for | Example |
|---|---|---|
| `release: patch` | A fix | `0.1.0` → `0.1.1` |
| `release: minor` | A new feature | `0.1.0` → `0.2.0` |
| `release: major` | A change that is not compatible with earlier versions | `0.1.0` → `1.0.0` |
| `release: skip` | A change that users do not get, for example documentation or CI | No release |

The **Release label** check fails if a pull request has no release label, or more than one. The check summary shows the version that the merge will release.

### Test the staging build

For each push to a pull request, GitHub Actions does these steps:

1. Runs the lint and the tests.
2. Publishes a staging build to [TestPyPI](https://test.pypi.org/project/docuquery/). The version is the next version plus a unique number, for example `0.1.1.dev12345678`.
3. Installs the staging build from TestPyPI, and runs `docuquery --version` and `docuquery --help`.

To try a staging build yourself, use the version from the **Publish to TestPyPI (staging)** job:

```bash
python3 -m venv /tmp/dq-staging
source /tmp/dq-staging/bin/activate
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ "docuquery==0.1.1.dev12345678"
```

A pull request with `release: skip` does not get a staging build.

> **NOTE:** A pull request from a fork does not get a staging build or the default label. GitHub does not give forks permission to publish or to add labels. A maintainer must add the label.

### Merge the pull request

When you merge a pull request into `main`, GitHub Actions does these steps:

1. Reads the release label of the pull request.
2. Calculates the next version from the latest `vX.Y.Z` tag.
3. Runs the lint and the tests again.
4. Publishes the version to [PyPI](https://pypi.org/project/docuquery/).
5. Creates the `vX.Y.Z` tag and a GitHub release with release notes.

If the label is `release: skip`, nothing is released.

> **NOTE:** The version in `pyproject.toml` sets only the first release, when there are no tags yet. After that, the version comes from the tags. Do not change the version in `pyproject.toml`.

> **NOTE:** Releases run one at a time. If you merge several pull requests while a release runs, GitHub keeps only the newest waiting release. The changes from the other merges are included in that release, with one version change.
