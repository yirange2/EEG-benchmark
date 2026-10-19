# Flanker

## 1. Overview

This is a PsychoPy adaptation of the ERP CORE Flankers ERN-LRP experiment. Participants respond to the direction of the center arrow while ignoring the surrounding arrows.

Practice and main are separate programs:

```bash
./psychopy_env/bin/python 04-Flanker/flanker_practice.py
./psychopy_env/bin/python 04-Flanker/flanker_main.py
```

Both dialogs ask for Participant ID and Run. Existing main data can be
overwritten after confirmation. Press `Escape` to stop.

Adapted from ERP CORE by Emily S. Kappenman and Steven J. Luck under CC BY-SA
4.0; see `ERP_CORE_LICENSE.txt`.

## 2. Experimental design

- Practice contains 10 trials and creates no data files.
- Main contains 10 blocks of 40 trials, for 400 trials in total.
- Every block contains 10 trials from each condition: congruent-left,
  congruent-right, incongruent-left, and incongruent-right.
- Each trial presents five arrows for 200 ms without a fixation point, followed by a white fixation point for a randomly sampled 1200–1400 ms interval.
- Participants press `LEFT` or `RIGHT` according to the center arrow. The
  response window begins at stimulus onset and ends with the following interval.
- After each block except the last, feedback is based on that block's error rate. The break includes an unskippable 20 s countdown, after which the participant may continue resting until pressing `LEFT` or `RIGHT`.

## 3. Triggers

Stimulus codes are sent on the arrow-onset flip:

| Condition | Stimulus | Code |
| --- | --- | ---: |
| Congruent left | `<<<<<` | 11 |
| Congruent right | `>>>>>` | 12 |
| Incongruent left | `>><>>` | 21 |
| Incongruent right | `<<><<` | 22 |

Response codes are:

| Condition | Left response | Right response |
| --- | ---: | ---: |
| Congruent left | 111 (correct) | 211 (incorrect) |
| Congruent right | 112 (incorrect) | 212 (correct) |
| Incongruent left | 121 (correct) | 221 (incorrect) |
| Incongruent right | 122 (incorrect) | 222 (correct) |

Each trigger is reset to 0 on the following display flip. Parallel-port output
is off by default:

```bash
./psychopy_env/bin/python 04-Flanker/flanker_main.py \
  --send-parallel-triggers --parallel-address 0x0378
```

## 4. Data structure

```text
04-Flanker/data/
├── dataset_description.json
├── participants.tsv
├── sourcedata/sub-001/
│   └── sub-001_task-flanker_run-01_desc-timing_metrics.json
└── sub-001/eeg/
    ├── sub-001_task-flanker_run-01_events.tsv
    └── sub-001_task-flanker_run-01_events.json
```

Each event row records block and trial number, condition, stimulus and response
codes, response key, accuracy, response time, requested interval, and measured
display timing. The timing file also contains block error rates and frame
diagnostics. Practice does not save data.

## 5. Display configuration

- Windowed presentation: 1100 × 700 px
- macOS scaled resolution: 1470 × 956 px
- Native panel resolution: 2560 × 1664 px
- Active display size: approximately 29.0 × 18.9 cm
- Viewing distance: 40 cm
- Refresh rate: fixed at 60 Hz; not measured by the program
- Retina rendering: enabled
- Background: RGB255 `(100, 100, 100)`
- Arrow height: 1.0° visual angle
- Fixation point: white, 0.15° diameter; shown only during the interval

Use the command-line display options shown by `--help` when moving the
experiment to another device.
