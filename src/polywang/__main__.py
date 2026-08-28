"""Allow `python -m polywang` and `uv run polywang` to start the scanner."""

from polywang.arbitrage_bot import main

if __name__ == "__main__":
    raise SystemExit(main())
