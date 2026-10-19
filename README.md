# EEG Benchmark Experiments

Four windowed PsychoPy experiments configured for a fixed 60 Hz display:

- `01-EOEC`: eyes-open/eyes-closed resting-state EEG
- `02-SSVEP`: four-target SSVEP task
- `03-Face`: ERP CORE N170 face-perception task
- `04-Flanker`: ERP CORE Flanker task

Each experiment folder contains its own design, trigger, data, and display
documentation.

## Installation

Use Python 3.11. Create the environment locally after cloning; do not copy the
original `psychopy_env` folder between computers.

macOS or Linux:

```bash
python3.11 -m venv psychopy_env
source psychopy_env/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv psychopy_env
psychopy_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The tested PsychoPy version is 2026.2.4.

## Running the experiments

Run commands from the repository root:

```bash
./psychopy_env/bin/python 01-EOEC/EOEC_main.py

./psychopy_env/bin/python 02-SSVEP/ssvep_practice.py
./psychopy_env/bin/python 02-SSVEP/ssvep_main.py

./psychopy_env/bin/python 03-Face/n170_practice.py
./psychopy_env/bin/python 03-Face/n170_main.py

./psychopy_env/bin/python 04-Flanker/flanker_practice.py
./psychopy_env/bin/python 04-Flanker/flanker_main.py
```

Participant data are written inside each experiment's local `data` directory.
All `data` directories are excluded from Git and must not be committed.

Parallel-port triggers are disabled by default. The correct trigger interface
depends on the EEG hardware; see each experiment README before enabling output.

## Validation

The design checks do not open experiment windows:

```bash
./psychopy_env/bin/python -m unittest discover -s 01-EOEC -p 'test_*.py'
./psychopy_env/bin/python -m unittest discover -s 02-SSVEP -p 'test_*.py'
./psychopy_env/bin/python -m unittest discover -s 03-Face -p 'test_*.py'
./psychopy_env/bin/python -m unittest discover -s 04-Flanker -p 'test_*.py'
```

## License

This repository is shared under CC BY-SA 4.0. The N170 and Flanker materials
include adaptations of ERP CORE by Emily S. Kappenman and Steven J. Luck. See
`LICENSE.md` and the experiment-specific `ERP_CORE_LICENSE.txt` files.
