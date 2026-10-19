#!/usr/bin/env python3
"""Run only the 320-trial ERP CORE N170 main program."""

import sys

from n170_task import main


if __name__ == "__main__":
    sys.exit(main(program_phase="main"))
