# SSVEP

## 1. Overview

This experiment presents four circles corresponding to 12 Hz, 15 Hz, 20 Hz,
and a non-flickering condition. Participants attend the cued circle and report
its location after stimulation. The paradigm is based on Li et al. (2016), with
the stimulus geometry and luminance waveform specified for this adaptation.

Practice and main are separate programs:

```bash
./psychopy_env/bin/python 02-SSVEP/ssvep_practice.py
./psychopy_env/bin/python 02-SSVEP/ssvep_main.py
```

Both dialogs ask for Participant ID and Run. Practice does not save data or send
triggers. Existing main data can be overwritten after confirmation.

## 2. Experimental design

- Practice contains four trials: one for each target location.
- Main contains 32 randomized trials in one continuous block: eight trials for
  each attended condition.
- Frequency-to-location assignment is fixed within a participant and balanced
  across participants with a four-group Latin square.

Each trial has the following structure:

1. The target circle turns red for 0.5 s.
2. All circles are white for a 0.5 s gaze-shift period.
3. The four circles are presented for 7 s. Three flicker at 12, 15, and 20 Hz;
   the fourth remains static.
4. A centered question asks which circle the participant looked at. The response
   is self-paced and uses the corresponding arrow key.
5. A central white cross is shown for a 4 s interval before the next trial.

## 3. Triggers

| Attended condition | Code |
| --- | ---: |
| 12 Hz | 12 |
| 15 Hz | 15 |
| 20 Hz | 20 |
| Static | 1 |

Main sends the condition code on the first stimulation frame and resets the
port to 0 on the following frame. Parallel-port output is off by default:

```bash
./psychopy_env/bin/python 02-SSVEP/ssvep_main.py \
  --send-parallel-triggers --parallel-address 0x0378
```

## 4. Data structure

```text
02-SSVEP/data/
├── dataset_description.json
├── participants.tsv
├── sourcedata/sub-001/
│   └── sub-001_task-ssvep_run-01_desc-timing_metrics.json
└── sub-001/eeg/
    ├── sub-001_task-ssvep_run-01_events.tsv
    └── sub-001_task-ssvep_run-01_events.json
```

Each event row records the attended frequency and location, trigger code,
stimulation timing, arrow-key response, response time, and response accuracy.
`participants.tsv` stores the participant's frequency-location counterbalance
group. Practice creates no data files.

## 5. Display configuration

- Windowed presentation: 1100 × 700 px
- macOS scaled resolution: 1470 × 956 px
- Active display size: approximately 29.0 × 18.9 cm
- Viewing distance: 40 cm
- Refresh rate: fixed at 60 Hz; not measured by the program
- Retina rendering: enabled
- Background: RGB255 `(100, 100, 100)`
- Circle diameter: 3.6° visual angle, approximately 2.51 cm at 40 cm
- Circle centers: 6.5° above, right, below, and left of screen center
- Flicker waveform: frame-based sinusoidal luminance modulation from background
  gray to white; the static circle remains white

Use the command-line display options shown by `--help` when moving the
experiment to another device.
