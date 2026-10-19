# N170 face perception

## 1. Overview

This is a PsychoPy adaptation of the ERP CORE N170 face-perception experiment.
Participants classify intact faces/cars as objects and scrambled images as textures. The original bitmap stimuli are stored locally in `03-Face/stimuli`.

Practice and main are separate programs:

```bash
./psychopy_env/bin/python 03-Face/n170_practice.py
./psychopy_env/bin/python 03-Face/n170_main.py
```

Both dialogs ask for Participant ID and Run. The default response mapping is `UP` for an intact object and `DOWN` for a texture. Existing main data can be overwritten after confirmation. Press `Escape` to stop.

Adapted from ERP CORE by Emily S. Kappenman and Steven J. Luck under CC BY-SA 4.0; see `ERP_CORE_LICENSE.txt`.

## 2. Experimental design

- Practice contains 10 trials and creates no data files.
- Main contains 320 trials divided into eight blocks of 40 trials.
- There are four conditions: faces, cars, scrambled faces, and scrambled cars. Every block contains exactly 10 trials per condition. Across the full run, each of 40 unique images per condition is shown twice.
- Each trial presents an image for 300 ms, followed by a white fixation point for a randomly sampled 1100–1300 ms interval.
- The response window begins at image onset and ends with the following interval.
- After each block except the last, there is an unskippable 20 s countdown. Participants may continue resting afterward and press either response key when ready. 

## 3. Triggers

| Condition | Stimulus codes |
| --- | --- |
| Faces | 1–40 |
| Cars | 41–80 |
| Scrambled faces | 101–140 |
| Scrambled cars | 141–180 |

| Response | Code |
| --- | ---: |
| Correct | 201 |
| Incorrect | 202 |

The stimulus code is sent on the image-onset flip. The first valid response is
also coded, and each code is reset to 0 on the following display flip.
Parallel-port output is off by default:

```bash
./psychopy_env/bin/python 03-Face/n170_main.py \
  --send-parallel-triggers --parallel-address 0x0378
```

## 4. Data structure

```text
03-Face/data/
├── dataset_description.json
├── participants.tsv
├── sourcedata/sub-001/
│   └── sub-001_task-n170_run-01_desc-timing_metrics.json
└── sub-001/eeg/
    ├── sub-001_task-n170_run-01_events.tsv
    └── sub-001_task-n170_run-01_events.json
```

Each event row records block and trial number, condition, stimulus file,
stimulus and response codes, response key, accuracy, response time, requested
interval, and measured display timing. The timing file summarizes frame
intervals and image durations. Practice does not save data.

## 5. Display configuration

- Windowed presentation: 1100 × 700 px
- macOS scaled resolution: 1470 × 956 px
- Native panel resolution: 2560 × 1664 px
- Active display size: approximately 29.0 × 18.9 cm
- Viewing distance: 40 cm
- Refresh rate: fixed at 60 Hz; not measured by the program
- Retina rendering: enabled
- Background: RGB255 `(100, 100, 100)`
- Image width: 3.32° visual angle, approximately 2.32 cm at 40 cm
- Fixation point: white, 0.15° diameter

Use the command-line display options shown by `--help` when moving the
experiment to another device.
