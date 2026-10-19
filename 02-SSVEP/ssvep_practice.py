#!/usr/bin/env python3
"""Run the four-trial SSVEP practice without triggers or saved data."""

import sys

from ssvep_task import main


if __name__ == "__main__":
    sys.exit(main(program_phase="practice"))
