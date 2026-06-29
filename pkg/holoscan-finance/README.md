# holoscan-finance package

This directory defines the Debian package for `holoscan-finance`.

## Usage

```bash
./holohub package holoscan-finance --pkg-generator DEB
```

## metadata.json

`metadata.json` registers this package with the holohub CLI. Two fields matter:

- **`package` key** — marks this directory as a HoloHub *package* project. The CLI
  discovers it via the recursive `HOLOSCAN_CLI_SEARCH_PATH` scan from the module root,
  which makes it appear under the `PACKAGES` section of `./holohub list`.
- **`package.dockerfile`** — path (relative to the module root) to the container
  definition used by `./holohub package` for container-based package builds.

Looking for Python packaging? Review the project [pyproject.toml](../../pyproject.toml)
for configuration.
