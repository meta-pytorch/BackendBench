# Contributing to BackendBench

## License

BackendBench is BSD-3-Clause licensed, as found in the LICENSE file.

## Our Development Process

BackendBench is actively developed internally at Meta and synced to GitHub regularly. External contributions are welcomed and will be reviewed by the Meta team.

## Code Quality

We use [ruff](https://docs.astral.sh/ruff/) for linting and code formatting.

## Pre-commit Hooks

To make development easier, we provide pre-commit hooks that automatically run ruff on your changes:

```bash
pip install pre-commit
pre-commit install
```

This will automatically lint your code before each commit, ensuring consistent code quality across the project.

## Pull Requests

We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")

In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues

We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.