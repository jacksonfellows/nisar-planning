## Quickstart

This project uses uv to manage Python dependencies.

Install uv:
- macOS/Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Windows (PowerShell): `irm https://astral.sh/uv/install.ps1 | iex`
- macOS (Homebrew): `brew install uv`

Install packages for this project by running `uv sync` in the project directory.

To see the available options for `main.py`:
```shell
uv run python main.py --help
```

To find tracks that intersect the test study area in a 12 hour period:
```shell
uv run python main.py test_boundary/boundary.shp '2025-11-05T20:00:00.000Z' test_intersections/intersections.shp --n-hours 12
```