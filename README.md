# EEG Benchmark Experiments

Four windowed PsychoPy experiments configured for a fixed 60 Hz display:

- `01-EOEC`: eyes-open/eyes-closed resting-state EEG
- `02-SSVEP`: four-target SSVEP task
- `03-Face`: ERP CORE N170 face-perception task
- `04-Flanker`: ERP CORE Flanker task

Each experiment folder contains its own design, trigger, data, and display documentation.

## Environment

Use Python 3.11. 
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

Parallel-port triggers are disabled by default. The correct trigger interface depends on the EEG hardware; see each experiment README before enabling output.
