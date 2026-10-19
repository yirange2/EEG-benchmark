#!/usr/bin/env python3
"""Run only the ERP CORE Flanker practice block."""

import sys

from flanker_task import main


if __name__ == "__main__":
    sys.exit(main(program_phase="practice"))
