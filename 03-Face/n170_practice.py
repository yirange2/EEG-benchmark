#!/usr/bin/env python3
"""Run only the 10-trial ERP CORE N170 practice program."""

import sys

from n170_task import main


if __name__ == "__main__":
    sys.exit(main(program_phase="practice"))
