#!/usr/bin/env python3
"""Four-target SSVEP task adapted from Li et al. (2016).

Run practice and main separately:
    python ssvep_practice.py
    python ssvep_main.py

The task uses a fixed 60 Hz timing model and never measures refresh rate.
Practice sends no EEG triggers and saves no data. Main writes BIDS-oriented
event and timing files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------

# Defaults for the current 13.6-inch MacBook Air configuration. Measure the
# active display width with a ruler before formal data collection and change
# --display-width-cm if needed.
DISPLAY_WIDTH_CM = 29.0
DISPLAY_HEIGHT_CM = 18.9
VIEWING_DISTANCE_CM = 40.0
DEFAULT_SCREEN_SIZE_PX = (1470, 956)
DEFAULT_WINDOW_SIZE_PX = (1100, 700)
NATIVE_SCREEN_SIZE_PX = (2560, 1664)
EXPECTED_REFRESH_HZ = 60.0


# ---------------------------------------------------------------------------
# Paradigm and stimulus configuration
# ---------------------------------------------------------------------------

CUE_DURATION_S = 0.5
GAZE_SHIFT_DURATION_S = 0.5
STIMULUS_DURATION_S = 7.0
ITI_DURATION_S = 4.0
READY_DURATION_S = 1.0

MAIN_TRIALS_PER_TARGET = 8
PRACTICE_TRIALS_PER_TARGET = 1

STIMULUS_SIZE_DEG = 3.6
STIMULUS_POSITIONS_DEG = {
    "up": (0.0, 6.5),
    "right": (6.5, 0.0),
    "down": (0.0, -6.5),
    "left": (-6.5, 0.0),
}

# Match the gray background used by the existing ERP tasks. Flickering circles
# move sinusoidally between the background gray and white. The non-flickering
# control circle remains white, as requested.
BACKGROUND_LUMINANCE_RGB255 = 100.0
BACKGROUND_RGB255 = (100, 100, 100)
FLICKER_MAX_LUMINANCE_RGB255 = 255.0
STATIC_LUMINANCE_RGB255 = 255.0
TEXT_RGB255 = (0, 0, 0)
CUE_TARGET_RGB255 = (255, 60, 60)

CENTRAL_FIXATION_HEIGHT_DEG = 0.6
INSTRUCTION_TEXT_HEIGHT_DEG = 0.65
READY_TEXT_HEIGHT_DEG = 0.72

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class Target:
    name: str
    location: str
    frequency_hz: float
    trigger_code: int

    @property
    def position(self) -> tuple[float, float]:
        return STIMULUS_POSITIONS_DEG[self.location]

    @property
    def trial_type(self) -> str:
        if self.frequency_hz == 0:
            return "static"
        return f"ssvep_{self.frequency_hz:g}hz"


@dataclass(frozen=True)
class Trial:
    phase: str
    trial_number: int
    target: Target


# Trigger code 0 is reserved for resetting a parallel port, so the static
# condition uses code 1. Frequency conditions use their frequency as the code.
LOCATION_ORDER = ("up", "right", "down", "left")
ATTENTION_RESPONSE_KEYS = LOCATION_ORDER
TARGET_CONDITIONS = (
    ("12 Hz", 12.0, 12),
    ("15 Hz", 15.0, 15),
    ("20 Hz", 20.0, 20),
    ("Static", 0.0, 1),
)
COUNTERBALANCE_GROUPS = 4


def targets_for_counterbalance_group(group: int) -> tuple[Target, ...]:
    """Return one row of the cyclic four-condition Latin square."""
    if not 1 <= group <= COUNTERBALANCE_GROUPS:
        raise ValueError(f"Counterbalance group must be 1-{COUNTERBALANCE_GROUPS}")

    offset = group - 1
    return tuple(
        Target(name, location, frequency_hz, trigger_code)
        for location_index, location in enumerate(LOCATION_ORDER)
        for name, frequency_hz, trigger_code in (
            (TARGET_CONDITIONS[(location_index + offset) % len(TARGET_CONDITIONS)]),
        )
    )


def counterbalance_group_for_participant(participant: str) -> int:
    """Assign a stable group from participant ID, independent of run number.

    Positive numeric labels cycle exactly across groups (001 -> 1, ..., 004 ->
    4, 005 -> 1). Other labels use a stable SHA-256-derived assignment.
    """
    if participant.isdigit() and int(participant) > 0:
        return ((int(participant) - 1) % COUNTERBALANCE_GROUPS) + 1
    digest = hashlib.sha256(participant.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % COUNTERBALANCE_GROUPS) + 1


def targets_for_participant(participant: str) -> tuple[int, tuple[Target, ...]]:
    group = counterbalance_group_for_participant(participant)
    return group, targets_for_counterbalance_group(group)


# Group 1 remains available as the default for validation and the smoke test.
TARGETS = targets_for_counterbalance_group(1)


def frames_for_seconds(
    duration_s: float,
    refresh_hz: float = EXPECTED_REFRESH_HZ,
) -> int:
    """Quantize a duration to the nearest whole video frame."""
    return max(1, int(round(duration_s * refresh_hz)))


def sine_luminance_rgb255(
    frequency_hz: float,
    frame_index: int,
    refresh_hz: float = EXPECTED_REFRESH_HZ,
) -> float:
    """Return frame luminance from 0 to 255 for cosine-phase sinusoidal flicker.

    Cosine is a sine wave shifted by pi/2. It starts at maximum luminance,
    creating an unambiguous physical change on the trigger-aligned first frame.
    A frequency of zero returns the temporal mean luminance.
    """
    if frequency_hz == 0:
        return STATIC_LUMINANCE_RGB255
    phase = 2.0 * math.pi * frequency_hz * frame_index / refresh_hz
    modulation = 0.5 * (1.0 + math.cos(phase))
    return BACKGROUND_LUMINANCE_RGB255 + (
        FLICKER_MAX_LUMINANCE_RGB255 - BACKGROUND_LUMINANCE_RGB255
    ) * modulation


def sanitize_bids_label(value: str, prefix: str, fallback: str) -> str:
    """Return an alphanumeric BIDS label without an entity prefix."""
    value = value.strip()
    value = re.sub(rf"^{re.escape(prefix)}-", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]+", "", value)
    return value or fallback


def build_trials(
    rng: random.Random,
    program_phase: str,
    targets: Iterable[Target] = TARGETS,
) -> list[Trial]:
    if program_phase not in {"practice", "main"}:
        raise ValueError(f"Unknown SSVEP phase: {program_phase}")

    repetitions = (
        PRACTICE_TRIALS_PER_TARGET
        if program_phase == "practice"
        else MAIN_TRIALS_PER_TARGET
    )
    target_tuple = tuple(targets)
    sequence = [target for target in target_tuple for _ in range(repetitions)]
    rng.shuffle(sequence)
    return [
        Trial(
            phase=program_phase.title(),
            trial_number=trial_number,
            target=target,
        )
        for trial_number, target in enumerate(sequence, start=1)
    ]


def validate_design(
    seed: int = 170,
    participant: str = "001",
) -> dict[str, object]:
    counterbalance_group, targets = targets_for_participant(participant)
    rng = random.Random(seed)
    practice = build_trials(rng, "practice", targets)
    main = build_trials(rng, "main", targets)

    condition_counts = {
        target.name: sum(trial.target == target for trial in main)
        for target in targets
    }
    cycle_samples = {
        target.name: (
            [STATIC_LUMINANCE_RGB255]
            if target.frequency_hz == 0
            else [
                sine_luminance_rgb255(target.frequency_hz, frame)
                for frame in range(
                    round(EXPECTED_REFRESH_HZ / target.frequency_hz)
                )
            ]
        )
        for target in targets
    }
    cycle_means = {
        name: statistics.mean(samples)
        for name, samples in cycle_samples.items()
    }

    report = {
        "seed": seed,
        "participant": participant,
        "counterbalance_group": counterbalance_group,
        "refresh_hz": EXPECTED_REFRESH_HZ,
        "practice_trials": len(practice),
        "main_trials": len(main),
        "main_trials_per_target": condition_counts,
        "timing_frames": {
            "cue": frames_for_seconds(CUE_DURATION_S),
            "gaze_shift": frames_for_seconds(GAZE_SHIFT_DURATION_S),
            "stimulation": frames_for_seconds(STIMULUS_DURATION_S),
            "iti": frames_for_seconds(ITI_DURATION_S),
        },
        "stimulus_size_deg": STIMULUS_SIZE_DEG,
        "stimulus_positions_deg": STIMULUS_POSITIONS_DEG,
        "frequency_location_mapping": {
            target.location: target.frequency_hz for target in targets
        },
        "all_counterbalance_mappings": {
            str(group): {
                target.location: target.frequency_hz
                for target in targets_for_counterbalance_group(group)
            }
            for group in range(1, COUNTERBALANCE_GROUPS + 1)
        },
        "trigger_codes": {
            target.name: target.trigger_code for target in targets
        },
        "attention_response_keys": ATTENTION_RESPONSE_KEYS,
        "attention_prompt_timing": "immediately_after_stimulation",
        "iti_after_attention_response": True,
        "one_cycle_luminance_rgb255": {
            name: [round(value, 3) for value in values]
            for name, values in cycle_samples.items()
        },
        "cycle_mean_luminance_rgb255": {
            name: round(value, 6) for name, value in cycle_means.items()
        },
        "display_defaults": {
            "screen_size_px": DEFAULT_SCREEN_SIZE_PX,
            "native_screen_size_px": NATIVE_SCREEN_SIZE_PX,
            "width_cm": DISPLAY_WIDTH_CM,
            "height_cm": DISPLAY_HEIGHT_CM,
            "viewing_distance_cm": VIEWING_DISTANCE_CM,
            "retina": True,
            "fullscreen": False,
        },
    }

    if len(practice) != 4 or len(main) != 32:
        raise AssertionError("Unexpected practice or main trial count")
    if set(condition_counts.values()) != {MAIN_TRIALS_PER_TARGET}:
        raise AssertionError(f"Unbalanced main design: {condition_counts}")
    flicker_means = [
        cycle_means[target.name] for target in targets if target.frequency_hz > 0
    ]
    if max(flicker_means) - min(flicker_means) > 1e-9:
        raise AssertionError(
            f"Flickering conditions have unequal temporal means: {cycle_means}"
        )
    if len({target.trigger_code for target in targets}) != len(targets):
        raise AssertionError("Trigger codes must be unique")
    return report


def ensure_bids_dataset_root(participant: str, counterbalance_group: int) -> None:
    """Create minimal dataset-level metadata without overwriting existing data."""
    DATA_DIR.mkdir(exist_ok=True)
    description_path = DATA_DIR / "dataset_description.json"
    if not description_path.exists():
        description_path.write_text(
            json.dumps(
                {
                    "Name": "Li 2016 SSVEP PsychoPy Adaptation",
                    "BIDSVersion": "1.11.2",
                    "DatasetType": "raw",
                    "Authors": [
                        "Guangli Li",
                        "Dan Zhang",
                        "Sizhe Wang",
                        "Yanwen Y. Duan",
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    participants_path = DATA_DIR / "participants.tsv"
    participant_id = f"sub-{participant}"
    if participants_path.exists():
        with participants_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            fieldnames = reader.fieldnames or []
            rows = list(reader)
        if "participant_id" not in fieldnames:
            raise RuntimeError(
                f"Existing {participants_path} has no participant_id column"
            )
        if "ssvep_counterbalance_group" not in fieldnames:
            fieldnames.append("ssvep_counterbalance_group")
            for row in rows:
                row["ssvep_counterbalance_group"] = "n/a"
    else:
        fieldnames = ["participant_id", "ssvep_counterbalance_group"]
        rows = []

    participant_row = next(
        (row for row in rows if row.get("participant_id") == participant_id),
        None,
    )
    if participant_row is None:
        participant_row = {name: "n/a" for name in fieldnames}
        participant_row["participant_id"] = participant_id
        rows.append(participant_row)
    participant_row["ssvep_counterbalance_group"] = str(counterbalance_group)

    with participants_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_visual_smoke_test() -> int:
    """Open a brief window and exercise the actual drawing path."""
    try:
        from psychopy import core, monitors, visual
    except ImportError as exc:
        raise RuntimeError("PsychoPy is required for the visual smoke test") from exc

    monitor = monitors.Monitor(
        "Li2016_SSVEP_smoke",
        width=DISPLAY_WIDTH_CM,
        distance=VIEWING_DISTANCE_CM,
        verbose=False,
        autoLog=False,
    )
    monitor.setSizePix((800, 600))
    win = visual.Window(
        size=(800, 600),
        fullscr=False,
        monitor=monitor,
        units="deg",
        color=BACKGROUND_RGB255,
        colorSpace="rgb255",
        waitBlanking=True,
        checkTiming=False,
        allowGUI=True,
    )
    try:
        circles = {
            target: visual.Circle(
                win,
                radius=STIMULUS_SIZE_DEG / 2.0,
                pos=target.position,
                units="deg",
                fillColor=(STATIC_LUMINANCE_RGB255,) * 3,
                lineColor=None,
                colorSpace="rgb255",
                edges=128,
            )
            for target in TARGETS
        }
        central_fixation = visual.TextStim(
            win,
            text="+",
            pos=(0.0, 0.0),
            units="deg",
            color=(255, 255, 255),
            colorSpace="rgb255",
            height=CENTRAL_FIXATION_HEIGHT_DEG,
            bold=False,
        )
        for frame in range(60):
            for target in TARGETS:
                luminance = sine_luminance_rgb255(target.frequency_hz, frame)
                circles[target].fillColor = (luminance,) * 3
                circles[target].draw()
            win.flip()
        for _ in range(12):
            central_fixation.draw()
            win.flip()
        core.wait(0.1)
    finally:
        win.close()
    print("SSVEP visual smoke test completed.")
    return 0


def run_experiment(args: argparse.Namespace, program_phase: str) -> int:
    if program_phase not in {"practice", "main"}:
        raise ValueError(f"Unknown SSVEP program phase: {program_phase}")

    try:
        from psychopy import core, event, gui, logging, monitors, visual
    except ImportError as exc:
        raise RuntimeError(
            "PsychoPy is required. Use ../psychopy_env/bin/python or install "
            "PsychoPy in the active environment."
        ) from exc

    dialog_values = {
        "Participant ID": args.participant,
        "Run": args.run,
    }
    dialog = gui.DlgFromDict(
        dialog_values,
        title=f"SSVEP - {program_phase.title()}",
        sortKeys=False,
    )
    if not dialog.OK:
        return 1

    participant = sanitize_bids_label(
        str(dialog_values["Participant ID"]),
        prefix="sub",
        fallback="anonymous",
    )
    run_id = sanitize_bids_label(
        str(dialog_values["Run"]),
        prefix="run",
        fallback="01",
    )
    if not run_id.isdigit():
        error_dialog = gui.Dlg(title="Invalid BIDS Run")
        error_dialog.addText("Run must contain digits only, for example 01.")
        error_dialog.show()
        return 1

    counterbalance_group, targets = targets_for_participant(participant)
    mapping_text = ", ".join(
        f"{target.location}={target.frequency_hz:g} Hz"
        if target.frequency_hz > 0
        else f"{target.location}=static"
        for target in targets
    )
    print(
        f"Participant {participant}: counterbalance group "
        f"{counterbalance_group} ({mapping_text})"
    )

    subject_label = f"sub-{participant}"
    if program_phase == "main":
        output_dir = DATA_DIR / subject_label / "eeg"
        output_stem = f"{subject_label}_task-ssvep_run-{run_id}"
        events_path = output_dir / f"{output_stem}_events.tsv"
        events_json_path = output_dir / f"{output_stem}_events.json"
        timing_dir = DATA_DIR / "sourcedata" / subject_label
        timing_path = timing_dir / f"{output_stem}_desc-timing_metrics.json"
        existing_outputs = [
            path
            for path in (events_path, events_json_path, timing_path)
            if path.exists()
        ]
        if existing_outputs:
            overwrite_dialog = gui.Dlg(title="Overwrite Existing Data?")
            overwrite_dialog.addText(
                "Data already exist for this Participant ID and Run:\n\n"
                + "\n".join(path.name for path in existing_outputs)
                + "\n\nPress OK to overwrite these files, or Cancel to return "
                "without starting the experiment."
            )
            overwrite_dialog.show()
            if not overwrite_dialog.OK:
                return 1

    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**32))
    rng = random.Random(seed)
    selected_trials = build_trials(rng, program_phase, targets)

    screen_size = (args.screen_width_px, args.screen_height_px)
    window_size = (args.window_width_px, args.window_height_px)
    monitor = monitors.Monitor(
        "Li2016_SSVEP",
        width=args.display_width_cm,
        distance=args.viewing_distance_cm,
        verbose=False,
        autoLog=False,
    )
    monitor.setSizePix(screen_size)

    win = visual.Window(
        size=window_size,
        fullscr=False,
        monitor=monitor,
        units="deg",
        color=BACKGROUND_RGB255,
        colorSpace="rgb255",
        waitBlanking=True,
        checkTiming=False,
        allowGUI=True,
        useRetina=True,
    )
    win.mouseVisible = True
    logging.console.setLevel(logging.WARNING)

    class TriggerPort:
        def __init__(self) -> None:
            self.port = None
            self.mode = "none"
            # Practice never sends triggers, even if the command-line flag is
            # accidentally supplied.
            if program_phase == "main" and args.send_parallel_triggers:
                from psychopy import parallel

                address = int(args.parallel_address, 0)
                self.port = parallel.ParallelPort(address=address)
                self.port.setData(0)
                self.mode = f"parallel:{hex(address)}"

        def send(self, code: int) -> None:
            if self.port is not None:
                self.port.setData(int(code))

        def reset(self) -> None:
            if self.port is not None:
                self.port.setData(0)

        def close(self) -> None:
            self.reset()

    trigger = TriggerPort()

    text_stim = visual.TextStim(
        win,
        text="",
        color=TEXT_RGB255,
        colorSpace="rgb255",
        font="Geneva",
        height=INSTRUCTION_TEXT_HEIGHT_DEG,
        wrapWidth=22,
        alignText="left",
        anchorHoriz="center",
        anchorVert="center",
    )

    circles = {
        target: visual.Circle(
            win,
            radius=STIMULUS_SIZE_DEG / 2.0,
            pos=target.position,
            units="deg",
            fillColor=(STATIC_LUMINANCE_RGB255,) * 3,
            lineColor=None,
            colorSpace="rgb255",
            edges=128,
        )
        for target in targets
    }
    central_fixation = visual.TextStim(
        win,
        text="+",
        pos=(0.0, 0.0),
        units="deg",
        color=(255, 255, 255),
        colorSpace="rgb255",
        height=CENTRAL_FIXATION_HEIGHT_DEG,
        bold=False,
    )

    instructions = (
        "Four circles will appear on the screen. In each trial, one of them "
        "will turn red.\n\n"
        "You must look at the cued circle throughout the 7-s stimulation and "
        "avoid blinking or moving.\n\n"
        "After the stimulation ends, report which circle you looked at using "
        "the UP, RIGHT, DOWN, or LEFT arrow key.\n\n"
        "Then, look at the central cross during the 4-s interval before the "
        "next trial starts.\n\n"
        "Press SPACE to begin."
    )

    def key_name(key: object) -> str:
        if isinstance(key, (tuple, list)):
            return str(key[0])
        return str(getattr(key, "name", key))

    def abort_if_requested(keys: Iterable[object]) -> None:
        if any(key_name(key) == "escape" for key in keys):
            raise KeyboardInterrupt

    def check_escape() -> None:
        abort_if_requested(event.getKeys(keyList=["escape"]))

    def show_message(message: str, align_text: str = "left") -> None:
        event.clearEvents(eventType="keyboard")
        text_stim.text = message
        text_stim.height = INSTRUCTION_TEXT_HEIGHT_DEG
        text_stim.alignText = align_text
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=["space", "escape"], clearEvents=True)
        abort_if_requested(keys or [])
        event.clearEvents(eventType="keyboard")

    def show_timed_text(message: str, frames: int) -> None:
        text_stim.text = message
        text_stim.height = READY_TEXT_HEIGHT_DEG
        text_stim.alignText = "center"
        for _ in range(frames):
            text_stim.draw()
            win.flip()
            check_escape()

    def ready_set_go() -> None:
        frames = frames_for_seconds(READY_DURATION_S)
        for word in ("Ready", "Set", "Go"):
            show_timed_text(word, frames)

    def draw_static_circles() -> None:
        for target in targets:
            circles[target].fillColor = (STATIC_LUMINANCE_RGB255,) * 3
            circles[target].draw()

    def draw_direction_cue(target_location: str) -> None:
        for target in targets:
            color = (
                CUE_TARGET_RGB255
                if target.location == target_location
                else (STATIC_LUMINANCE_RGB255,) * 3
            )
            circles[target].fillColor = color
            circles[target].draw()

    data_rows: list[dict[str, object]] = []
    experiment_clock = core.Clock()
    response_clock = core.Clock()
    experiment_status = "completed"

    def mark_time(target: dict[str, float], key: str) -> None:
        target[key] = experiment_clock.getTime()

    def collect_attention_response(
        trial: Trial,
        flip_times: dict[str, float],
    ) -> dict[str, object]:
        event.clearEvents(eventType="keyboard")
        text_stim.text = "Which circle did you look at?"
        text_stim.height = INSTRUCTION_TEXT_HEIGHT_DEG
        text_stim.alignText = "center"
        text_stim.draw()
        win.callOnFlip(response_clock.reset)
        win.callOnFlip(event.clearEvents, eventType="keyboard")
        win.callOnFlip(mark_time, flip_times, "stimulus_offset")
        win.callOnFlip(mark_time, flip_times, "response_prompt_onset")
        win.flip()
        keys = event.waitKeys(
            keyList=[*ATTENTION_RESPONSE_KEYS, "escape"],
            timeStamped=response_clock,
            clearEvents=True,
        )
        abort_if_requested(keys or [])
        response_key, response_rt_s = (keys or [("", "")])[0]
        reported_location = str(response_key)
        return {
            "attention_response_key": reported_location,
            "attention_reported_location": reported_location,
            "attention_response_rt_s": float(response_rt_s),
            "attention_response_accuracy": int(
                reported_location == trial.target.location
            ),
            "attention_prompt_onset_s": flip_times["response_prompt_onset"],
        }

    def present_trial(trial: Trial) -> None:
        cue_frames = frames_for_seconds(CUE_DURATION_S)
        gaze_frames = frames_for_seconds(GAZE_SHIFT_DURATION_S)
        stimulus_frames = frames_for_seconds(STIMULUS_DURATION_S)
        iti_frames = frames_for_seconds(ITI_DURATION_S)
        flip_times: dict[str, float] = {}

        win.recordFrameIntervals = True

        for frame in range(cue_frames):
            draw_direction_cue(trial.target.location)
            if frame == 0:
                win.callOnFlip(mark_time, flip_times, "cue_onset")
            win.flip()
            check_escape()

        for frame in range(gaze_frames):
            draw_static_circles()
            if frame == 0:
                win.callOnFlip(mark_time, flip_times, "gaze_shift_onset")
            win.flip()
            check_escape()

        for frame in range(stimulus_frames):
            for display_target in targets:
                luminance = sine_luminance_rgb255(
                    display_target.frequency_hz,
                    frame,
                )
                circles[display_target].fillColor = (luminance,) * 3
                circles[display_target].draw()
            if frame == 0:
                win.callOnFlip(trigger.send, trial.target.trigger_code)
                win.callOnFlip(mark_time, flip_times, "stimulus_onset")
            elif frame == 1:
                win.callOnFlip(trigger.reset)
            win.flip()
            check_escape()

        win.recordFrameIntervals = False

        attention_response = collect_attention_response(trial, flip_times)

        win.recordFrameIntervals = True
        for frame in range(iti_frames):
            central_fixation.draw()
            if frame == 0:
                win.callOnFlip(mark_time, flip_times, "iti_onset")
            if frame == iti_frames - 1:
                win.callOnFlip(mark_time, flip_times, "iti_last_flip")
            win.flip()
            check_escape()
        win.recordFrameIntervals = False

        stimulus_onset = flip_times["stimulus_onset"]
        stimulus_offset = flip_times["stimulus_offset"]
        data_rows.append(
            {
                "participant": participant,
                "run": run_id,
                "counterbalance_group": counterbalance_group,
                "seed": seed,
                "phase": trial.phase,
                "trial_number": trial.trial_number,
                "target_name": trial.target.name,
                "target_location": trial.target.location,
                "target_x_deg": trial.target.position[0],
                "target_y_deg": trial.target.position[1],
                "target_frequency_hz": trial.target.frequency_hz,
                "stimulus_code": trial.target.trigger_code,
                "cue_onset_s": flip_times["cue_onset"],
                "gaze_shift_onset_s": flip_times["gaze_shift_onset"],
                "stimulus_onset_s": stimulus_onset,
                "stimulus_offset_s": stimulus_offset,
                "iti_onset_s": flip_times["iti_onset"],
                "iti_last_flip_s": flip_times["iti_last_flip"],
                "measured_cue_ms": (
                    flip_times["gaze_shift_onset"] - flip_times["cue_onset"]
                )
                * 1000.0,
                "measured_gaze_shift_ms": (
                    stimulus_onset - flip_times["gaze_shift_onset"]
                )
                * 1000.0,
                "measured_stimulus_ms": (
                    stimulus_offset - stimulus_onset
                )
                * 1000.0,
                "cue_frames": cue_frames,
                "gaze_shift_frames": gaze_frames,
                "stimulus_frames": stimulus_frames,
                "iti_frames": iti_frames,
                **attention_response,
            }
        )

    runtime_error: Optional[Exception] = None
    win.recordFrameIntervals = False
    try:
        experiment_clock.reset()
        if program_phase == "practice":
            show_message(
                instructions
                + "\n\nThis practice contains one trial for each target."
            )
        else:
            show_message(instructions)
        ready_set_go()

        for trial in selected_trials:
            present_trial(trial)

        if program_phase == "practice":
            show_message(
                "Practice complete.\n\nPlease wait for the experimenter.",
                align_text="center",
            )
        else:
            show_message(
                "You have completed the experiment.\n\nThank you for participating.",
                align_text="center",
            )
    except KeyboardInterrupt:
        experiment_status = "interrupted"
        if program_phase == "practice":
            print("Practice interrupted. No practice data were saved.")
        else:
            print("Experiment interrupted. Saving completed trial data.")
    except Exception as exc:
        experiment_status = "error"
        runtime_error = exc
        if program_phase == "practice":
            print(f"Practice error: {exc}. No practice data were saved.")
        else:
            print(f"Experiment error: {exc}. Saving completed trial data.")
    finally:
        elapsed_s = experiment_clock.getTime()
        frame_intervals = list(win.frameIntervals)
        trigger.close()
        win.close()

    if program_phase == "practice":
        if experiment_status == "completed":
            print("Practice finished. No practice data were saved and no triggers were sent.")
        if runtime_error is not None:
            raise runtime_error
        return 0 if experiment_status == "completed" else 130

    ensure_bids_dataset_root(participant, counterbalance_group)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    first_onset_s = next(
        (
            float(row["stimulus_onset_s"])
            for row in data_rows
            if isinstance(row.get("stimulus_onset_s"), (int, float))
        ),
        0.0,
    )
    event_fieldnames = [
        "onset",
        "duration",
        "trial_type",
        "value",
        "trial_number",
        "counterbalance_group",
        "target_frequency_hz",
        "target_location",
        "response_time",
        "response_key",
        "reported_location",
        "response_accuracy",
        "response_prompt_onset",
        "target_x_deg",
        "target_y_deg",
        "cue_onset",
        "gaze_shift_onset",
        "stimulus_size_deg",
        "waveform",
        "refresh_rate_hz",
        "cue_frames",
        "gaze_shift_frames",
        "stimulus_frames",
        "iti_frames",
        "presentation_onset_s",
        "presentation_offset_s",
        "measured_cue_ms",
        "measured_gaze_shift_ms",
        "measured_stimulus_ms",
        "random_seed",
    ]
    event_rows: list[dict[str, object]] = []
    for row in data_rows:
        target = next(
            target for target in targets if target.name == row["target_name"]
        )
        onset = float(row["stimulus_onset_s"]) - first_onset_s
        event_rows.append(
            {
                "onset": onset,
                "duration": float(row["measured_stimulus_ms"]) / 1000.0,
                "trial_type": target.trial_type,
                "value": row["stimulus_code"],
                "trial_number": row["trial_number"],
                "counterbalance_group": row["counterbalance_group"],
                "target_frequency_hz": row["target_frequency_hz"],
                "target_location": row["target_location"],
                "response_time": row["attention_response_rt_s"],
                "response_key": row["attention_response_key"],
                "reported_location": row["attention_reported_location"],
                "response_accuracy": row["attention_response_accuracy"],
                "response_prompt_onset": (
                    float(row["attention_prompt_onset_s"]) - first_onset_s
                ),
                "target_x_deg": row["target_x_deg"],
                "target_y_deg": row["target_y_deg"],
                "cue_onset": float(row["cue_onset_s"]) - first_onset_s,
                "gaze_shift_onset": (
                    float(row["gaze_shift_onset_s"]) - first_onset_s
                ),
                "stimulus_size_deg": STIMULUS_SIZE_DEG,
                "waveform": "sampled_sinusoidal_luminance_cosine_phase",
                "refresh_rate_hz": EXPECTED_REFRESH_HZ,
                "cue_frames": row["cue_frames"],
                "gaze_shift_frames": row["gaze_shift_frames"],
                "stimulus_frames": row["stimulus_frames"],
                "iti_frames": row["iti_frames"],
                "presentation_onset_s": row["stimulus_onset_s"],
                "presentation_offset_s": row["stimulus_offset_s"],
                "measured_cue_ms": row["measured_cue_ms"],
                "measured_gaze_shift_ms": row["measured_gaze_shift_ms"],
                "measured_stimulus_ms": row["measured_stimulus_ms"],
                "random_seed": row["seed"],
            }
        )

    with events_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=event_fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(event_rows)

    events_metadata = {
        "onset": {
            "Description": "SSVEP stimulation onset relative to the first stimulation onset",
            "Units": "s",
        },
        "duration": {
            "Description": "Measured stimulation duration from PsychoPy flip timestamps",
            "Units": "s",
        },
        "trial_type": {
            "Description": "Attended target condition",
            "Levels": {
                "ssvep_12hz": "Attend the circle flickering at 12 Hz",
                "ssvep_15hz": "Attend the circle flickering at 15 Hz",
                "ssvep_20hz": "Attend the circle flickering at 20 Hz",
                "static": "Attend the non-flickering circle",
            },
        },
        "value": {
            "Description": "EEG trigger value sent on the stimulation-onset flip",
        },
        "counterbalance_group": {
            "Description": "Participant-level Latin-square frequency-to-location mapping group",
            "Levels": {
                str(group): ", ".join(
                    f"{target.location}={target.frequency_hz:g} Hz"
                    if target.frequency_hz > 0
                    else f"{target.location}=static"
                    for target in targets_for_counterbalance_group(group)
                )
                for group in range(1, COUNTERBALANCE_GROUPS + 1)
            },
        },
        "target_frequency_hz": {
            "Description": "Temporal frequency of the attended target; 0 denotes static",
            "Units": "Hz",
        },
        "target_location": {
            "Description": "Location of the attended target under the participant-level counterbalancing scheme",
            "Levels": {
                "up": "Upper circle",
                "right": "Right circle",
                "down": "Lower circle",
                "left": "Left circle",
            },
        },
        "response_time": {
            "Description": "Arrow-key response time from onset of the post-trial attention-check prompt",
            "Units": "s",
        },
        "response_key": {
            "Description": "Arrow key pressed for the post-trial attention check",
            "Levels": {
                "up": "Upper circle",
                "right": "Right circle",
                "down": "Lower circle",
                "left": "Left circle",
            },
        },
        "reported_location": {
            "Description": "Circle location the participant reported looking at",
        },
        "response_accuracy": {
            "Description": "Whether the reported location matched the cued target",
            "Levels": {
                "0": "Reported location did not match the cue",
                "1": "Reported location matched the cue",
            },
        },
        "response_prompt_onset": {
            "Description": "Attention-check prompt onset relative to the first stimulation onset",
            "Units": "s",
        },
        "cue_onset": {
            "Description": "Red target-circle cue onset relative to the first stimulation onset",
            "Units": "s",
        },
        "gaze_shift_onset": {
            "Description": "Onset of the white-circle gaze-shift interval",
            "Units": "s",
        },
        "stimulus_size_deg": {
            "Description": "Diameter of each circle",
            "Units": "degree",
        },
        "waveform": {
            "Description": "Frame-based luminance modulation method",
        },
        "presentation_onset_s": {
            "Description": "Raw PsychoPy experiment-clock timestamp",
            "Units": "s",
        },
        "presentation_offset_s": {
            "Description": "Raw PsychoPy experiment-clock timestamp",
            "Units": "s",
        },
        "measured_stimulus_ms": {
            "Description": "Stimulation duration from onset and offset flip timestamps",
            "Units": "ms",
        },
    }
    events_json_path.write_text(
        json.dumps(events_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    measured_stimulus_ms = [
        float(row["measured_stimulus_ms"]) for row in data_rows
    ]
    measured_cue_ms = [float(row["measured_cue_ms"]) for row in data_rows]
    measured_gaze_ms = [
        float(row["measured_gaze_shift_ms"]) for row in data_rows
    ]
    attention_accuracy = [
        int(row["attention_response_accuracy"]) for row in data_rows
    ]
    attention_response_times = [
        float(row["attention_response_rt_s"]) for row in data_rows
    ]

    def timing_stats(values: list[float], expected_ms: float) -> dict[str, object]:
        if not values:
            return {
                "n": 0,
                "expected_ms": expected_ms,
                "mean_ms": None,
                "median_ms": None,
                "minimum_ms": None,
                "maximum_ms": None,
                "mean_absolute_error_ms": None,
            }
        return {
            "n": len(values),
            "expected_ms": expected_ms,
            "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
            "mean_absolute_error_ms": statistics.mean(
                abs(value - expected_ms) for value in values
            ),
        }

    expected_interval_s = 1.0 / EXPECTED_REFRESH_HZ
    dropped_threshold_s = 1.5 * expected_interval_s
    summary = {
        "participant": participant,
        "run": run_id,
        "counterbalance_group": counterbalance_group,
        "task": "ssvep",
        "phase": program_phase,
        "status": experiment_status,
        "random_seed": seed,
        "completed_trials": len(data_rows),
        "planned_trials": len(selected_trials),
        "elapsed_s": elapsed_s,
        "expected_refresh_hz": EXPECTED_REFRESH_HZ,
        "refresh_rate_measured": False,
        "trigger_mode": trigger.mode,
        "display": {
            "screen_size_px": list(screen_size),
            "window_size_px": list(window_size),
            "fullscreen": False,
            "display_width_cm": args.display_width_cm,
            "display_height_cm": args.display_height_cm,
            "viewing_distance_cm": args.viewing_distance_cm,
            "retina": True,
        },
        "paradigm": {
            "stimulus_size_deg": STIMULUS_SIZE_DEG,
            "stimulus_positions_deg": STIMULUS_POSITIONS_DEG,
            "frequency_location_mapping": {
                target.location: target.frequency_hz for target in targets
            },
            "trigger_codes": {
                target.name: target.trigger_code for target in targets
            },
            "waveform": "sampled_sinusoidal_luminance_cosine_phase",
            "stimulus_shape": "circle",
            "background_rgb255": list(BACKGROUND_RGB255),
            "flicker_luminance_range_rgb255": [
                BACKGROUND_LUMINANCE_RGB255,
                FLICKER_MAX_LUMINANCE_RGB255,
            ],
            "static_luminance_rgb255": STATIC_LUMINANCE_RGB255,
            "cue_duration_s": CUE_DURATION_S,
            "gaze_shift_duration_s": GAZE_SHIFT_DURATION_S,
            "stimulation_duration_s": STIMULUS_DURATION_S,
            "iti_duration_s": ITI_DURATION_S,
            "post_trial_attention_check": True,
            "attention_prompt_timing": "immediately_after_stimulation",
            "iti_after_attention_response": True,
            "attention_response_keys": list(ATTENTION_RESPONSE_KEYS),
        },
        "attention_check": {
            "completed_responses": len(attention_accuracy),
            "correct_responses": sum(attention_accuracy),
            "accuracy": (
                statistics.mean(attention_accuracy)
                if attention_accuracy
                else None
            ),
            "mean_response_time_s": (
                statistics.mean(attention_response_times)
                if attention_response_times
                else None
            ),
        },
        "timing_from_flip_timestamps": {
            "cue": timing_stats(measured_cue_ms, CUE_DURATION_S * 1000.0),
            "gaze_shift": timing_stats(
                measured_gaze_ms,
                GAZE_SHIFT_DURATION_S * 1000.0,
            ),
            "stimulation": timing_stats(
                measured_stimulus_ms,
                STIMULUS_DURATION_S * 1000.0,
            ),
        },
        "frame_interval_diagnostics": {
            "recorded_intervals": len(frame_intervals),
            "expected_interval_ms": expected_interval_s * 1000.0,
            "dropped_frame_threshold_ms": dropped_threshold_s * 1000.0,
            "intervals_over_threshold": sum(
                interval > dropped_threshold_s for interval in frame_intervals
            ),
            "maximum_interval_ms": (
                max(frame_intervals) * 1000.0 if frame_intervals else None
            ),
            "mean_interval_ms": (
                statistics.mean(frame_intervals) * 1000.0
                if frame_intervals
                else None
            ),
        },
        "note": (
            "Flip timestamps verify software scheduling, not emitted light. "
            "Use a photodiode for end-to-end display and trigger validation."
        ),
    }
    timing_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved events: {events_path}")
    print(f"Saved events metadata: {events_json_path}")
    print(f"Saved timing diagnostics: {timing_path}")
    if measured_stimulus_ms:
        print(
            "Stimulation timing from flip timestamps: "
            f"mean={statistics.mean(measured_stimulus_ms):.3f} ms, "
            f"min={min(measured_stimulus_ms):.3f} ms, "
            f"max={max(measured_stimulus_ms):.3f} ms"
        )

    if runtime_error is not None:
        raise runtime_error
    return 0 if experiment_status == "completed" else 130


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the design without opening PsychoPy",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="briefly draw the stimuli in a PsychoPy window",
    )
    parser.add_argument("--participant", default="debug", help="participant identifier")
    parser.add_argument(
        "--run",
        default="01",
        help="numeric BIDS run label",
    )
    parser.add_argument(
        "--window-width-px",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PX[0],
    )
    parser.add_argument(
        "--window-height-px",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PX[1],
    )
    parser.add_argument(
        "--screen-width-px",
        type=int,
        default=DEFAULT_SCREEN_SIZE_PX[0],
    )
    parser.add_argument(
        "--screen-height-px",
        type=int,
        default=DEFAULT_SCREEN_SIZE_PX[1],
    )
    parser.add_argument(
        "--display-width-cm",
        type=float,
        default=DISPLAY_WIDTH_CM,
    )
    parser.add_argument(
        "--display-height-cm",
        type=float,
        default=DISPLAY_HEIGHT_CM,
        help="recorded in timing metadata; degree conversion uses width",
    )
    parser.add_argument(
        "--viewing-distance-cm",
        type=float,
        default=VIEWING_DISTANCE_CM,
    )
    parser.add_argument(
        "--send-parallel-triggers",
        action="store_true",
        help="main only: send one-frame parallel-port onset triggers",
    )
    parser.add_argument(
        "--parallel-address",
        default="0x0378",
        help="parallel-port address parsed with Python base-0 syntax",
    )
    parser.add_argument("--seed", type=int, default=None, help="fixed random seed")
    return parser.parse_args(argv)


def main(
    argv: Optional[list[str]] = None,
    program_phase: str = "main",
) -> int:
    args = parse_args(argv)
    if args.validate_only:
        validation_seed = args.seed if args.seed is not None else 170
        participant = sanitize_bids_label(
            args.participant,
            prefix="sub",
            fallback="anonymous",
        )
        print(
            json.dumps(
                validate_design(validation_seed, participant=participant),
                indent=2,
            )
        )
        return 0
    if args.smoke_test:
        return run_visual_smoke_test()
    return run_experiment(args, program_phase=program_phase)


if __name__ == "__main__":
    sys.exit(main())
