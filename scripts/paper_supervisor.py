#!/usr/bin/env python3
"""QUANT-20260909-01: paper supervisor CLI.

Start (from the repo root, secrets stay in the environment / local .env):

    uv run python scripts/paper_supervisor.py --markets 200 --cash 1000

Clean stop without a respawn loop — pick one, then wait for the child to exit
(or send SIGTERM to the supervisor):

    touch paper-supervisor.stop
    # or: PAPER_SUPERVISOR_STOP=1

See docs/LIVE_RUNBOOK.md.
"""

from polywang.paper_supervisor import main

if __name__ == "__main__":
    raise SystemExit(main())
