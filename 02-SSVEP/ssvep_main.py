#!/usr/bin/env python3
"""Run the 32-trial Li et al. SSVEP main experiment."""

import sys

from ssvep_task import main


if __name__ == "__main__":
    sys.exit(main(program_phase="main"))
