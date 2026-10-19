# Releasing

Releases are **tag-driven**. Pushing a `v*` tag builds the package, creates a
GitHub Release, and publishes to PyPI. Nothing publishes from a normal merge to
`main`.

## One-time setup (already done)

- Repository variable `PYPI_PUBLISH = true`.
- Repository secret `PYPI_API_TOKEN` (a PyPI API token). Alternatively you can
  delete the secret and instead add a PyPI Trusted Publisher for
  `veritas-research` → owner `Nyvo2010`, repo `veritas`, workflow `release.yml`,
  environment `pypi`.

## Cutting a release

All of this happens through a pull request, because `main` is protected.

1. **Bump the version and note the change.**

   ```bash
   git checkout main && git pull
   git checkout -b release-v1.1.0
   ```

   - `pyproject.toml`: set `version = "1.1.0"`.
   - `CHANGELOG.md`: add a `## [1.1.0]` section.

2. **Open a PR into `main`**, wait for the `tests` checks to pass, and merge.

3. **Tag the merged commit.**

   ```bash
   git checkout main && git pull
   git tag -a v1.1.0 -m "Veritas v1.1.0"
   git push origin v1.1.0
   ```

4. **Watch the release workflow.** `Actions → release` runs `build` (wheel +
   sdist, smoke test, GitHub Release) and `pypi` (publishes to
   https://pypi.org/project/veritas-research/).

   ```bash
   gh run watch "$(gh run list --workflow=release.yml --limit 1 --json databaseId -q '.[0].databaseId')"
   ```

## Rules

- The tag **must** match `pyproject.toml`'s version (`v1.1.0` ↔ `1.1.0`).
- A version already on PyPI can never be re-uploaded. To fix a bad release,
  cut a new patch version; do not move the tag.
- The version in `pyproject.toml` on `main` is the *next* version to ship. After
  releasing, the next PR bumps it again.
