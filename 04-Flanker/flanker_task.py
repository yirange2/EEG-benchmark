#!/usr/bin/env python3
"""PsychoPy adaptation of the ERP CORE Flankers ERN-LRP task.

Run practice and main as separate programs:
    python flanker_practice.py
    python flanker_main.py

The defaults reproduce the ERP CORE 11.20.18 experiment parameters while
using the same windowed, fixed-60-Hz MacBook configuration as the N170 port.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional


# Local display configuration shared with the N170 adaptation.
DISPLAY_WIDTH_CM = 29.0
DISPLAY_HEIGHT_CM = 18.9
VIEWING_DISTANCE_CM = 40.0
DEFAULT_SCREEN_SIZE_PX = (1470, 956)
DEFAULT_WINDOW_SIZE_PX = (1100, 700)
NATIVE_SCREEN_SIZE_PX = (2560, 1664)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# ERP CORE Flankers ERN-LRP defaults.
EXPECTED_REFRESH_HZ = 60.0
TARGET_DURATION_MS = 200
ISI_MIN_MS = 1200
ISI_MAX_MS = 1400
READY_DURATION_MS = 1000
MINIMUM_BREAK_DURATION_S = 20
BLOCKS = 10
TRIALS_PER_CONDITION_PER_BLOCK = 10
PRACTICE_TRIALS = 10
MINIMUM_ERROR_RATE = 0.10
MAXIMUM_ERROR_RATE = 0.20

BACKGROUND_RGB255 = (100, 100, 100)
TEXT_RGB255 = (0, 0, 0)
FIXATION_RGB255 = (255, 255, 255)
FIXATION_DIAMETER_DEG = 0.15
STIMULUS_HEIGHT_DEG = 1.0
INSTRUCTION_TEXT_HEIGHT_DEG = 0.72
READY_TEXT_HEIGHT_DEG = 0.72


@dataclass(frozen=True)
class Condition:
    name: str
    compatibility: str
    target_direction: str
    center_character: str
    flanker_character: str
    stimulus_text: str
    trigger_code: int
    correct_key: str


@dataclass(frozen=True)
class Trial:
    phase: str
    block_number: int
    trial_number: int
    sequence_number: int
    condition: Condition


@dataclass(frozen=True)
class KeyResponse:
    name: str
    rt: float


CONDITIONS = (
    Condition(
        name="Congruent Left",
        compatibility="Compatible",
        target_direction="Left",
        center_character="<",
        flanker_character="<",
        stimulus_text="<<<<<",
        trigger_code=11,
        correct_key="left",
    ),
    Condition(
        name="Congruent Right",
        compatibility="Compatible",
        target_direction="Right",
        center_character=">",
        flanker_character=">",
        stimulus_text=">>>>>",
        trigger_code=12,
        correct_key="right",
    ),
    Condition(
        name="Incongruent Left",
        compatibility="Incompatible",
        target_direction="Left",
        center_character="<",
        flanker_character=">",
        stimulus_text=">><>>",
        trigger_code=21,
        correct_key="left",
    ),
    Condition(
        name="Incongruent Right",
        compatibility="Incompatible",
        target_direction="Right",
        center_character=">",
        flanker_character="<",
        stimulus_text="<<><<",
        trigger_code=22,
        correct_key="right",
    ),
)


def frames_for_ms(duration_ms: int, refresh_hz: float = EXPECTED_REFRESH_HZ) -> int:
    """Quantize a requested duration to the nearest whole video frame."""
    return max(1, int(round(duration_ms * refresh_hz / 1000.0)))


def sanitize_bids_label(value: str, prefix: str, fallback: str) -> str:
    """Return an alphanumeric BIDS label without an entity prefix."""
    value = value.strip()
    value = re.sub(rf"^{re.escape(prefix)}-", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]+", "", value)
    return value or fallback


def response_trigger_code(response_key: str, stimulus_code: int) -> int:
    """Match Presentation's response-code construction for this task."""
    if response_key not in {"left", "right"}:
        raise ValueError(f"Unknown response key: {response_key}")
    button_prefix = 1 if response_key == "left" else 2
    return int(f"{button_prefix}{stimulus_code}")


def condition_pool() -> list[Condition]:
    return [
        condition
        for condition in CONDITIONS
        for _ in range(TRIALS_PER_CONDITION_PER_BLOCK)
    ]


def build_main_trials(rng: random.Random) -> list[Trial]:
    """Build 10 independently shuffled, balanced 40-trial blocks."""
    trials: list[Trial] = []
    sequence_number = 0
    for block_number in range(1, BLOCKS + 1):
        block_conditions = condition_pool()
        rng.shuffle(block_conditions)
        for trial_number, condition in enumerate(block_conditions, start=1):
            sequence_number += 1
            trials.append(
                Trial(
                    phase="Main",
                    block_number=block_number,
                    trial_number=trial_number,
                    sequence_number=sequence_number,
                    condition=condition,
                )
            )
    return trials


def build_practice_trials(rng: random.Random) -> list[Trial]:
    """Sample 10 trials with replacement, then shuffle as in Presentation."""
    source = condition_pool()
    selected = [rng.choice(source) for _ in range(PRACTICE_TRIALS)]
    rng.shuffle(selected)
    return [
        Trial(
            phase="Practice",
            block_number=0,
            trial_number=number,
            sequence_number=number,
            condition=condition,
        )
        for number, condition in enumerate(selected, start=1)
    ]


def validate_design(seed: int = 170) -> dict[str, object]:
    rng = random.Random(seed)
    practice = build_practice_trials(rng)
    main = build_main_trials(rng)

    total_condition_counts = {
        condition.name: sum(trial.condition == condition for trial in main)
        for condition in CONDITIONS
    }
    per_block_counts = {
        block: {
            condition.name: sum(
                trial.block_number == block and trial.condition == condition
                for trial in main
            )
            for condition in CONDITIONS
        }
        for block in range(1, BLOCKS + 1)
    }
    response_codes = {
        condition.name: {
            "left": response_trigger_code("left", condition.trigger_code),
            "right": response_trigger_code("right", condition.trigger_code),
        }
        for condition in CONDITIONS
    }

    if len(practice) != PRACTICE_TRIALS:
        raise AssertionError("Unexpected practice trial count")
    if len(main) != BLOCKS * len(CONDITIONS) * TRIALS_PER_CONDITION_PER_BLOCK:
        raise AssertionError("Unexpected main trial count")
    if any(
        count != TRIALS_PER_CONDITION_PER_BLOCK
        for block_counts in per_block_counts.values()
        for count in block_counts.values()
    ):
        raise AssertionError("A main block is not balanced")

    return {
        "seed": seed,
        "refresh_hz": EXPECTED_REFRESH_HZ,
        "target_frames": frames_for_ms(TARGET_DURATION_MS),
        "target_actual_ms": (
            frames_for_ms(TARGET_DURATION_MS) / EXPECTED_REFRESH_HZ * 1000.0
        ),
        "isi_frame_range": [frames_for_ms(ISI_MIN_MS), frames_for_ms(ISI_MAX_MS)],
        "ready_frames": frames_for_ms(READY_DURATION_MS),
        "minimum_break_duration_s": MINIMUM_BREAK_DURATION_S,
        "break_can_extend_until_ready": True,
        "fixation_during_stimulus": False,
        "fixation_during_isi": True,
        "blocks": BLOCKS,
        "trials_per_block": len(CONDITIONS) * TRIALS_PER_CONDITION_PER_BLOCK,
        "practice_trials": len(practice),
        "main_trials": len(main),
        "condition_trial_counts": total_condition_counts,
        "stimulus_trigger_codes": {
            condition.name: condition.trigger_code for condition in CONDITIONS
        },
        "response_trigger_codes": response_codes,
        "display_defaults": {
            "screen_size_px": DEFAULT_SCREEN_SIZE_PX,
            "native_screen_size_px": NATIVE_SCREEN_SIZE_PX,
            "width_cm": DISPLAY_WIDTH_CM,
            "height_cm": DISPLAY_HEIGHT_CM,
            "viewing_distance_cm": VIEWING_DISTANCE_CM,
            "retina": True,
        },
    }


def ensure_bids_dataset_root(participant: str) -> None:
    """Create minimal dataset-level metadata without overwriting it."""
    DATA_DIR.mkdir(exist_ok=True)
    description_path = DATA_DIR / "dataset_description.json"
    if not description_path.exists():
        description_path.write_text(
            json.dumps(
                {
                    "Name": "ERP CORE Flankers ERN-LRP PsychoPy",
                    "BIDSVersion": "1.11.2",
                    "DatasetType": "raw",
                    "Authors": ["Emily S. Kappenman", "Steven J. Luck"],
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
        if any(row.get("participant_id") == participant_id for row in rows):
            return
    else:
        fieldnames = ["participant_id"]
        rows = []

    rows.append(
        {
            name: participant_id if name == "participant_id" else "n/a"
            for name in fieldnames
        }
    )
    with participants_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def feedback_for_error_rate(error_rate: float) -> str:
    if error_rate < MINIMUM_ERROR_RATE:
        return "Try to respond a bit faster."
    if error_rate > MAXIMUM_ERROR_RATE:
        return "Try to respond more accurately."
    return "Great job!"


def run_visual_smoke_test() -> int:
    """Exercise the actual PsychoPy drawing path in a short window."""
    try:
        from psychopy import monitors, visual
    except ImportError as exc:
        raise RuntimeError("PsychoPy is required for the visual smoke test") from exc

    monitor = monitors.Monitor(
        "ERP_CORE_Flanker_smoke",
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
        allowGUI=True,
    )
    fixation = visual.Circle(
        win,
        radius=FIXATION_DIAMETER_DEG / 2.0,
        fillColor=FIXATION_RGB255,
        lineColor=FIXATION_RGB255,
        colorSpace="rgb255",
        edges=64,
    )
    arrows = visual.TextStim(
        win,
        text=CONDITIONS[2].stimulus_text,
        color=TEXT_RGB255,
        colorSpace="rgb255",
        font="Geneva",
        height=STIMULUS_HEIGHT_DEG,
    )
    try:
        for _ in range(frames_for_ms(TARGET_DURATION_MS)):
            arrows.draw()
            win.flip()
        for _ in range(12):
            fixation.draw()
            win.flip()
    finally:
        win.close()
    print("Flanker visual smoke test completed.")
    return 0


def run_experiment(args: argparse.Namespace, program_phase: str) -> int:
    if program_phase not in {"practice", "main"}:
        raise ValueError(f"Unknown Flanker program phase: {program_phase}")

    try:
        from psychopy import core, event, gui, logging, monitors, visual
    except ImportError as exc:
        raise RuntimeError(
            "PsychoPy is required to run the task. Use ../psychopy_env/bin/python "
            "or install PsychoPy in the active environment."
        ) from exc

    dialog_values = {
        "Participant ID": args.participant,
        "Run": args.run,
    }
    dialog = gui.DlgFromDict(
        dialog_values,
        title=f"ERP CORE Flanker - {program_phase.title()}",
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

    subject_label = f"sub-{participant}"
    if program_phase == "main":
        output_dir = DATA_DIR / subject_label / "eeg"
        output_stem = f"{subject_label}_task-flanker_run-{run_id}"
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
    selected_trials = (
        build_practice_trials(rng)
        if program_phase == "practice"
        else build_main_trials(rng)
    )

    screen_size = (args.screen_width_px, args.screen_height_px)
    window_size = (args.window_width_px, args.window_height_px)
    monitor = monitors.Monitor(
        "ERP_CORE_Flanker",
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
    response_clock = core.Clock()

    class TriggerPort:
        def __init__(self) -> None:
            self.port = None
            self.mode = "none"
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
    logging.console.setLevel(logging.WARNING)

    text_stim = visual.TextStim(
        win,
        text="",
        color=TEXT_RGB255,
        colorSpace="rgb255",
        font="Geneva",
        height=INSTRUCTION_TEXT_HEIGHT_DEG,
        wrapWidth=24,
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
    )
    arrow_stim = visual.TextStim(
        win,
        text="",
        color=TEXT_RGB255,
        colorSpace="rgb255",
        font="Geneva",
        height=STIMULUS_HEIGHT_DEG,
        alignText="center",
        anchorHoriz="center",
        anchorVert="center",
    )
    fixation = visual.Circle(
        win,
        radius=FIXATION_DIAMETER_DEG / 2.0,
        fillColor=FIXATION_RGB255,
        lineColor=FIXATION_RGB255,
        colorSpace="rgb255",
        edges=64,
    )

    instructions = (
        "Respond to the direction of the center arrow.\n\n"
        "Center arrow points left (<): press LEFT.\n"
        "Center arrow points right (>): press RIGHT.\n\n"
        "Respond quickly and accurately."
    )

    def key_name(key: object) -> str:
        if isinstance(key, (tuple, list)):
            return str(key[0])
        return str(getattr(key, "name", key))

    def abort_if_requested(keys: Iterable[object]) -> None:
        if any(key_name(key) == "escape" for key in keys):
            raise KeyboardInterrupt

    def show_message(message: str, align_text: str = "center") -> None:
        event.clearEvents(eventType="keyboard")
        text_stim.text = message
        text_stim.height = INSTRUCTION_TEXT_HEIGHT_DEG
        text_stim.alignText = align_text
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(
            keyList=["left", "right", "escape"],
            clearEvents=True,
        )
        abort_if_requested(keys or [])
        event.clearEvents(eventType="keyboard")

    def show_timed_text(message: str, frames: int) -> None:
        text_stim.text = message
        text_stim.height = READY_TEXT_HEIGHT_DEG
        text_stim.alignText = "center"
        for _ in range(frames):
            text_stim.draw()
            win.flip()
            abort_if_requested(event.getKeys(keyList=["escape"]))

    def ready_set_go() -> None:
        frames = frames_for_ms(READY_DURATION_MS)
        for word in ("Ready", "Set", "Go"):
            show_timed_text(word, frames)

    def show_break_countdown(feedback: str) -> None:
        event.clearEvents(eventType="keyboard")
        frames_per_second = frames_for_ms(1000)
        text_stim.height = INSTRUCTION_TEXT_HEIGHT_DEG
        text_stim.alignText = "center"
        for seconds_remaining in range(MINIMUM_BREAK_DURATION_S, 0, -1):
            text_stim.text = (
                "Take a rest!\n\n"
                f"{feedback}\n\n"
                f"Minimum rest: {seconds_remaining} seconds remaining."
            )
            for _ in range(frames_per_second):
                text_stim.draw()
                win.flip()
                abort_if_requested(event.getKeys(keyList=["escape"]))
        event.clearEvents(eventType="keyboard")
        show_message(
            "The minimum break is complete.\n\n"
            f"{feedback}\n\n"
            "Continue resting if needed. Press LEFT or RIGHT when you are "
            "ready to continue."
        )

    def show_fixation(frames: int) -> None:
        for _ in range(frames):
            fixation.draw()
            win.flip()
            abort_if_requested(event.getKeys(keyList=["escape"]))

    def collect_first_response(
        response: Optional[KeyResponse],
    ) -> Optional[KeyResponse]:
        keys = event.getKeys(
            keyList=["left", "right", "escape"],
            timeStamped=response_clock,
        )
        abort_if_requested(keys)
        if response is None:
            task_keys = [(name, rt) for name, rt in keys if name in {"left", "right"}]
            if task_keys:
                name, rt = min(task_keys, key=lambda key: key[1])
                return KeyResponse(name=name, rt=float(rt))
        return response

    def send_new_response_trigger(
        previous_response: Optional[KeyResponse],
        response: Optional[KeyResponse],
        stimulus_code: int,
    ) -> None:
        if previous_response is None and response is not None:
            trigger.send(response_trigger_code(response.name, stimulus_code))
            win.callOnFlip(trigger.reset)

    data_rows: list[dict[str, object]] = []
    experiment_status = "completed"
    experiment_clock = core.Clock()

    def mark_experiment_time(target: dict[str, float], key: str) -> None:
        target[key] = experiment_clock.getTime()

    def present_trial(trial: Trial) -> None:
        condition = trial.condition
        requested_isi_ms = rng.randint(ISI_MIN_MS, ISI_MAX_MS)
        isi_frames = frames_for_ms(requested_isi_ms)
        target_frames = frames_for_ms(TARGET_DURATION_MS)
        response: Optional[KeyResponse] = None
        arrow_stim.text = condition.stimulus_text

        win.recordFrameIntervals = True
        flip_times: dict[str, float] = {}
        for frame in range(target_frames):
            arrow_stim.draw()
            if frame == 0:
                win.callOnFlip(response_clock.reset)
                win.callOnFlip(event.clearEvents, eventType="keyboard")
                win.callOnFlip(trigger.send, condition.trigger_code)
                win.callOnFlip(mark_experiment_time, flip_times, "onset")
            elif frame == 1:
                win.callOnFlip(trigger.reset)
            win.flip()
            previous_response = response
            response = collect_first_response(response)
            send_new_response_trigger(
                previous_response,
                response,
                condition.trigger_code,
            )

        for frame in range(isi_frames):
            fixation.draw()
            if frame == 0:
                win.callOnFlip(mark_experiment_time, flip_times, "offset")
            win.flip()
            previous_response = response
            response = collect_first_response(response)
            send_new_response_trigger(
                previous_response,
                response,
                condition.trigger_code,
            )
        win.recordFrameIntervals = False

        if response is None:
            response_key: object = ""
            response_rt_s: object = ""
            accuracy = 0
            response_code: object = ""
        else:
            response_key = response.name
            response_rt_s = response.rt
            accuracy = int(response.name == condition.correct_key)
            response_code = response_trigger_code(
                response.name,
                condition.trigger_code,
            )

        data_rows.append(
            {
                "participant": participant,
                "run": run_id,
                "seed": seed,
                "phase": trial.phase,
                "block_number": trial.block_number,
                "trial_number": trial.trial_number,
                "sequence_number": trial.sequence_number,
                "condition": condition.name,
                "compatibility": condition.compatibility,
                "target_direction": condition.target_direction,
                "center_character": condition.center_character,
                "flanker_character": condition.flanker_character,
                "number_of_flankers_each_side": 2,
                "stimulus_text": condition.stimulus_text,
                "stimulus_code": condition.trigger_code,
                "correct_key": condition.correct_key,
                "response_key": response_key,
                "accuracy": accuracy,
                "rt_s": response_rt_s,
                "response_code": response_code,
                "requested_isi_ms": requested_isi_ms,
                "isi_frames": isi_frames,
                "frame_quantized_isi_ms": (
                    isi_frames / EXPECTED_REFRESH_HZ * 1000.0
                ),
                "stimulus_frames": target_frames,
                "stimulus_onset_s": flip_times.get("onset", ""),
                "stimulus_offset_s": flip_times.get("offset", ""),
                "measured_stimulus_ms": (
                    (flip_times["offset"] - flip_times["onset"]) * 1000.0
                    if "onset" in flip_times and "offset" in flip_times
                    else ""
                ),
            }
        )

    runtime_error: Optional[Exception] = None
    win.recordFrameIntervals = False
    try:
        experiment_clock.reset()

        if program_phase == "practice":
            show_message(
                instructions
                + "\n\nPress either response button to begin the practice trials.",
                align_text="left",
            )
            ready_set_go()
            show_fixation(frames_for_ms(rng.randint(ISI_MIN_MS, ISI_MAX_MS)))
            for trial in selected_trials:
                present_trial(trial)
            show_message(
                "Practice trials complete!\n\nPlease wait for the experimenter."
            )
        else:
            trials_by_block = {
                block: [
                    trial
                    for trial in selected_trials
                    if trial.block_number == block
                ]
                for block in range(1, BLOCKS + 1)
            }
            for block_number in range(1, BLOCKS + 1):
                if block_number == 1:
                    show_message(
                        instructions + "\n\nPress either button to begin.",
                        align_text="left",
                    )
                ready_set_go()
                show_fixation(frames_for_ms(rng.randint(ISI_MIN_MS, ISI_MAX_MS)))

                block_start = len(data_rows)
                for trial in trials_by_block[block_number]:
                    present_trial(trial)

                if block_number < BLOCKS:
                    block_rows = data_rows[block_start:]
                    error_rate = 1.0 - (
                        sum(int(row["accuracy"]) for row in block_rows)
                        / len(block_rows)
                    )
                    show_break_countdown(feedback_for_error_rate(error_rate))

            show_message(
                "You have now completed the experiment.\n\n"
                "Thank you for participating."
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
            print("Practice finished. No practice data were saved.")
        if runtime_error is not None:
            raise runtime_error
        return 0 if experiment_status == "completed" else 130

    ensure_bids_dataset_root(participant)
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
        "response_time",
        "stimulus",
        "value",
        "block_number",
        "trial_number",
        "sequence_number",
        "compatibility",
        "target_direction",
        "center_character",
        "flanker_character",
        "correct_response",
        "response",
        "accuracy",
        "response_value",
        "requested_isi_ms",
        "isi_frames",
        "frame_quantized_isi_ms",
        "stimulus_frames",
        "presentation_onset_s",
        "presentation_offset_s",
        "measured_stimulus_ms",
        "random_seed",
    ]
    event_rows: list[dict[str, object]] = []
    for row in data_rows:
        presentation_onset = float(row["stimulus_onset_s"])
        measured_duration_ms = float(row["measured_stimulus_ms"])
        event_rows.append(
            {
                "onset": presentation_onset - first_onset_s,
                "duration": measured_duration_ms / 1000.0,
                "trial_type": re.sub(
                    r"[^a-z0-9]+",
                    "_",
                    str(row["condition"]).lower(),
                ).strip("_"),
                "response_time": row["rt_s"] if row["rt_s"] != "" else "n/a",
                "stimulus": row["stimulus_text"],
                "value": row["stimulus_code"],
                "block_number": row["block_number"],
                "trial_number": row["trial_number"],
                "sequence_number": row["sequence_number"],
                "compatibility": row["compatibility"],
                "target_direction": row["target_direction"],
                "center_character": row["center_character"],
                "flanker_character": row["flanker_character"],
                "correct_response": row["correct_key"],
                "response": row["response_key"] if row["response_key"] != "" else "n/a",
                "accuracy": row["accuracy"],
                "response_value": (
                    row["response_code"] if row["response_code"] != "" else "n/a"
                ),
                "requested_isi_ms": row["requested_isi_ms"],
                "isi_frames": row["isi_frames"],
                "frame_quantized_isi_ms": row["frame_quantized_isi_ms"],
                "stimulus_frames": row["stimulus_frames"],
                "presentation_onset_s": row["stimulus_onset_s"],
                "presentation_offset_s": row["stimulus_offset_s"],
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
            "Description": (
                "Target onset relative to the first target in the PsychoPy log. "
                "Align to EEG acquisition using the recorded trigger values."
            ),
            "Units": "s",
        },
        "duration": {
            "Description": "Measured target/flanker presentation duration.",
            "Units": "s",
        },
        "trial_type": {"Description": "Compatibility and target direction."},
        "response_time": {
            "Description": "First response time relative to target onset.",
            "Units": "s",
        },
        "stimulus": {"Description": "Five displayed arrowhead characters."},
        "value": {"Description": "ERP CORE target trigger code."},
        "block_number": {"Description": "Main block number, 1 through 10."},
        "trial_number": {"Description": "Trial number within the block."},
        "sequence_number": {"Description": "Trial number across the run."},
        "compatibility": {"Description": "Compatible or Incompatible flankers."},
        "target_direction": {"Description": "Direction of the central target."},
        "center_character": {"Description": "Central target character."},
        "flanker_character": {"Description": "Flanker character."},
        "correct_response": {"Description": "Correct arrow-key response."},
        "response": {"Description": "First recorded arrow-key response."},
        "accuracy": {"Description": "Response accuracy, coded 0 or 1."},
        "response_value": {"Description": "ERP CORE response trigger code."},
        "requested_isi_ms": {"Description": "Randomly requested ISI.", "Units": "ms"},
        "isi_frames": {"Description": "ISI duration in 60 Hz video frames."},
        "frame_quantized_isi_ms": {
            "Description": "ISI after conversion to whole 60 Hz frames.",
            "Units": "ms",
        },
        "stimulus_frames": {"Description": "Target duration in 60 Hz frames."},
        "presentation_onset_s": {
            "Description": "Raw PsychoPy experiment-clock target-onset timestamp.",
            "Units": "s",
        },
        "presentation_offset_s": {
            "Description": "Raw PsychoPy experiment-clock target-offset timestamp.",
            "Units": "s",
        },
        "measured_stimulus_ms": {
            "Description": "Target duration measured from flip timestamps.",
            "Units": "ms",
        },
        "random_seed": {"Description": "Random seed used to construct this run."},
    }
    events_json_path.write_text(
        json.dumps(events_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    target_durations_ms = [
        float(row["measured_stimulus_ms"])
        for row in data_rows
        if isinstance(row.get("measured_stimulus_ms"), (int, float))
    ]
    one_frame_ms = 1000.0 / EXPECTED_REFRESH_HZ
    target_timing = {
        "target_ms": TARGET_DURATION_MS,
        "samples": len(target_durations_ms),
        "mean_ms": statistics.fmean(target_durations_ms) if target_durations_ms else None,
        "median_ms": statistics.median(target_durations_ms) if target_durations_ms else None,
        "minimum_ms": min(target_durations_ms) if target_durations_ms else None,
        "maximum_ms": max(target_durations_ms) if target_durations_ms else None,
        "mean_absolute_error_ms": (
            statistics.fmean(
                abs(duration - TARGET_DURATION_MS)
                for duration in target_durations_ms
            )
            if target_durations_ms
            else None
        ),
        "within_one_60hz_frame": sum(
            abs(duration - TARGET_DURATION_MS) <= one_frame_ms
            for duration in target_durations_ms
        ),
    }
    dropped_threshold_s = 1.5 / EXPECTED_REFRESH_HZ
    block_error_rates = {
        str(block): 1.0
        - statistics.fmean(
            float(row["accuracy"])
            for row in data_rows
            if int(row["block_number"]) == block
        )
        for block in range(1, BLOCKS + 1)
        if any(int(row["block_number"]) == block for row in data_rows)
    }
    summary = {
        "participant": participant,
        "run": run_id,
        "program_phase": program_phase,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "events_file": str(events_path.relative_to(DATA_DIR)),
        "seed": seed,
        "status": experiment_status,
        "error": repr(runtime_error) if runtime_error is not None else None,
        "assumed_refresh_hz": EXPECTED_REFRESH_HZ,
        "trigger_mode": trigger.mode,
        "completed_trials": len(data_rows),
        "planned_trials": len(selected_trials),
        "minimum_break_duration_s": MINIMUM_BREAK_DURATION_S,
        "break_can_extend_until_ready": True,
        "fixation_during_stimulus": False,
        "fixation_during_isi": True,
        "elapsed_s": elapsed_s,
        "block_error_rates": block_error_rates,
        "recorded_frame_intervals": len(frame_intervals),
        "dropped_frame_intervals": sum(
            interval > dropped_threshold_s for interval in frame_intervals
        ),
        "longest_frame_interval_ms": (
            max(frame_intervals) * 1000.0 if frame_intervals else None
        ),
        "target_timing_from_flip_timestamps": target_timing,
        "display": {
            "window_size_px": window_size,
            "screen_size_px": screen_size,
            "native_screen_size_px": NATIVE_SCREEN_SIZE_PX,
            "fullscreen": False,
            "retina": True,
            "width_cm": args.display_width_cm,
            "height_cm": args.display_height_cm,
            "viewing_distance_cm": args.viewing_distance_cm,
        },
    }
    timing_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved events: {events_path}")
    print(f"Saved events metadata: {events_json_path}")
    print(f"Saved timing diagnostics: {timing_path}")
    if target_durations_ms:
        print(
            "Target timing from flip timestamps: "
            f"mean={target_timing['mean_ms']:.3f} ms, "
            f"min={target_timing['minimum_ms']:.3f} ms, "
            f"max={target_timing['maximum_ms']:.3f} ms"
        )
    if runtime_error is not None:
        raise runtime_error
    return 0 if experiment_status == "completed" else 130


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check trial construction without opening PsychoPy",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="open a short window to test stimulus rendering, then exit",
    )
    parser.add_argument("--participant", default="debug", help="participant identifier")
    parser.add_argument(
        "--run",
        "--session",
        dest="run",
        default="01",
        metavar="RUN",
        help="run identifier used to prefill the startup dialog",
    )
    parser.add_argument(
        "--window-width-px",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PX[0],
        help="debug window width in pixels",
    )
    parser.add_argument(
        "--window-height-px",
        type=int,
        default=DEFAULT_WINDOW_SIZE_PX[1],
        help="debug window height in pixels",
    )
    parser.add_argument(
        "--screen-width-px",
        type=int,
        default=DEFAULT_SCREEN_SIZE_PX[0],
        help="display width reported to PsychoPy",
    )
    parser.add_argument(
        "--screen-height-px",
        type=int,
        default=DEFAULT_SCREEN_SIZE_PX[1],
        help="display height reported to PsychoPy",
    )
    parser.add_argument(
        "--display-width-cm",
        type=float,
        default=DISPLAY_WIDTH_CM,
        help="physical display width used for visual-angle conversion",
    )
    parser.add_argument(
        "--display-height-cm",
        type=float,
        default=DISPLAY_HEIGHT_CM,
        help="physical display height recorded in the timing summary",
    )
    parser.add_argument(
        "--viewing-distance-cm",
        type=float,
        default=VIEWING_DISTANCE_CM,
        help="viewing distance used for visual-angle conversion",
    )
    parser.add_argument(
        "--send-parallel-triggers",
        action="store_true",
        help="send main-task EEG event codes through a parallel port",
    )
    parser.add_argument(
        "--parallel-address",
        default="0x0378",
        help="parallel-port address, for example 0x0378",
    )
    parser.add_argument("--seed", type=int, default=None, help="fixed random seed")
    return parser.parse_args(argv)


def main(
    argv: Optional[list[str]] = None,
    program_phase: Optional[str] = None,
) -> int:
    args = parse_args(argv)
    if args.validate_only:
        report = validate_design(seed=args.seed if args.seed is not None else 170)
        print(json.dumps(report, indent=2))
        return 0
    if args.smoke_test:
        return run_visual_smoke_test()
    if program_phase is None:
        print(
            "Choose one program: flanker_practice.py for practice or "
            "flanker_main.py for the main experiment."
        )
        return 2
    return run_experiment(args, program_phase)


if __name__ == "__main__":
    sys.exit(main())
