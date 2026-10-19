# EOEC

## 1. Overview

This is an eyes-open/eyes-closed resting-state EEG experiment. 

Run it from the project root:

```bash
./psychopy_env/bin/python 01-EOEC/EOEC_main.py
```

The startup dialog asks for Participant ID and Run. Before recording, the two tones can be replayed with `R`; press `SPACE` to continue. Existing data can be overwritten after confirmation. Press `Escape` to stop the experiment.

## 2. Experimental design

- One run contains eight 60-second blocks: four eyes-open and four eyes-closed.
- There are no trials within a block and no gap between consecutive blocks.
- Conditions alternate throughout the run. Starting condition and tone mapping are balanced across four participant groups and remain fixed for a participant.
- The two signals are a 1000 Hz high tone and a 400 Hz low tone, each lasting 300 ms. One means eyes open and the other means eyes closed.
- Eyes-open blocks show an empty gray screen without a fixation point.


## 3. Triggers

| Condition | Code |
| --- | ---: |
| Eyes open | 11 |
| Eyes closed | 12 |

The tone, trigger, and block-onset timestamp are scheduled on the same display flip. The trigger is reset to 0 on the following frame. Parallel-port output is off by default and can be enabled with:

```bash
./psychopy_env/bin/python 01-EOEC/EOEC_main.py \
  --send-parallel-triggers --parallel-address 0x0378
```

## 4. Data structure

```text
01-EOEC/data/
├── dataset_description.json
├── participants.tsv
├── sourcedata/sub-001/
│   └── sub-001_task-eoec_run-01_desc-timing_metrics.json
└── sub-001/eeg/
    ├── sub-001_task-eoec_run-01_events.tsv
    └── sub-001_task-eoec_run-01_events.json
```

The event file contains one row per completed block, including condition,
onset, duration, trigger code, and the recommended analysis onset and duration.
The timing file contains display-flip timing and frame-interval diagnostics.

## 5. Display configuration

- Windowed presentation: 1100 × 700 px
- macOS scaled resolution: 1470 × 956 px
- Refresh rate: fixed at 60 Hz; not measured by the program
- Retina rendering: enabled
- Background: RGB255 `(100, 100, 100)`
- Text: black
- Fixation point: none
