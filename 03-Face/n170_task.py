#!/usr/bin/env python3
"""PsychoPy adaptation of the ERP CORE Face Perception N170 task.

The task design and stimuli are adapted from ERP CORE by Emily S. Kappenman
and Steven J. Luck. See ERP_CORE_LICENSE.txt in this directory.

Run a structural check without opening a PsychoPy window:
    python n170_task.py --validate-only --seed 170

Run practice and main as separate programs:
    python n170_practice.py
    python n170_main.py

The task starts directly in an 1100 x 700 window. Runtime settings can be
changed with command-line options; use --help to list them.
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
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

# MacBook Air 13-inch M4 configuration used for local development. PsychoPy's
# Retina mode expects the current macOS scaled resolution for degree-to-screen
# conversion; the panel's native resolution is 2560 x 1664 at 224 ppi.
DISPLAY_WIDTH_CM = 29.0
DISPLAY_HEIGHT_CM = 18.9
VIEWING_DISTANCE_CM = 40.0
DEFAULT_SCREEN_SIZE_PX = (1470, 956)
DEFAULT_WINDOW_SIZE_PX = (1100, 700)
NATIVE_SCREEN_SIZE_PX = (2560, 1664)

BASE_DIR = Path(__file__).resolve().parent
STIMULUS_DIR = BASE_DIR / "stimuli"
DATA_DIR = BASE_DIR / "data"

# Original ERP CORE defaults.
EXPECTED_REFRESH_HZ = 60.0
STIMULUS_DURATION_MS = 300
ISI_MIN_MS = 1100
ISI_MAX_MS = 1300
READY_DURATION_MS = 1000
MINIMUM_BREAK_DURATION_S = 20
BLOCKS = 8
TRIALS_PER_CONDITION_PER_BLOCK = 10
TRIALS_PER_CONDITION = BLOCKS * TRIALS_PER_CONDITION_PER_BLOCK
PRACTICE_TRIALS = 10
TRIALS_BETWEEN_BREAKS = 4 * TRIALS_PER_CONDITION_PER_BLOCK


BACKGROUND_RGB255 = (100, 100, 100)
TEXT_RGB255 = (0, 0, 0)
FIXATION_RGB255 = (255, 255, 255)
FIXATION_DIAMETER_DEG = 0.15
IMAGE_WIDTH_DEG = 3.32
IMAGE_PIXEL_SIZE = (184, 210)
IMAGE_HEIGHT_DEG = IMAGE_WIDTH_DEG * IMAGE_PIXEL_SIZE[1] / IMAGE_PIXEL_SIZE[0]
INSTRUCTION_TEXT_HEIGHT_DEG = 0.72
READY_TEXT_HEIGHT_DEG = 0.72

CORRECT_RESPONSE_CODE = 201
INCORRECT_RESPONSE_CODE = 202


@dataclass(frozen=True)
class Condition:
    type_name: str
    condition_name: str
    directory: str
    condition_index: int
    port_offset: int
    practice_names: tuple[str, ...]


@dataclass(frozen=True)
class Trial:
    phase: str
    block_number: int
    trial_number: int
    type_name: str
    condition_name: str
    stimulus_path: Path
    stimulus_index: int
    port_code: int
    correct_key: str


@dataclass(frozen=True)
class KeyResponse:
    name: str
    rt: float


CONDITIONS = (
    Condition(
        type_name="Object",
        condition_name="Faces",
        directory="faces",
        condition_index=1,
        port_offset=0,
        practice_names=("back_59.bmp", "back_60.bmp", "back_61.bmp"),
    ),
    Condition(
        type_name="Object",
        condition_name="Cars",
        directory="cars",
        condition_index=2,
        port_offset=40,
        practice_names=("back_V41.bmp", "back_V42.bmp", "back_V43.bmp"),
    ),
    Condition(
        type_name="Scrambled",
        condition_name="Scrambled Faces",
        directory="faces scrambled",
        condition_index=1,
        port_offset=100,
        practice_names=("S_back_59.bmp", "S_back_60.bmp", "S_back_61.bmp"),
    ),
    Condition(
        type_name="Scrambled",
        condition_name="Scrambled Cars",
        directory="cars scrambled",
        condition_index=2,
        port_offset=140,
        practice_names=("S_back_V41.bmp", "S_back_V42.bmp", "S_back_V43.bmp"),
    ),
)


def natural_key(path: Path) -> list[object]:
    """Sort filenames by embedded integers instead of lexicographically."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def condition_files(condition: Condition) -> list[Path]:
    folder = STIMULUS_DIR / condition.directory
    practice = set(condition.practice_names)
    return sorted(
        (path for path in folder.glob("*.bmp") if path.name not in practice),
        key=natural_key,
    )


def practice_files(condition: Condition) -> list[Path]:
    folder = STIMULUS_DIR / condition.directory
    return [folder / name for name in condition.practice_names]


def stimulus_code(condition: Condition, stimulus_index: int) -> int:
    """Reproduce ERP CORE's N170 trigger-code calculation."""
    return condition.port_offset + stimulus_index


def frames_for_ms(duration_ms: int, refresh_hz: float = EXPECTED_REFRESH_HZ) -> int:
    """Quantize a requested duration to the nearest whole video frame."""
    return max(1, int(round(duration_ms * refresh_hz / 1000.0)))


def validate_stimuli() -> dict[str, int]:
    counts: dict[str, int] = {}
    errors: list[str] = []

    for condition in CONDITIONS:
        main = condition_files(condition)
        practice = practice_files(condition)
        counts[condition.condition_name] = len(main)

        if len(main) != 40:
            errors.append(
                f"{condition.condition_name}: expected 40 main images, found {len(main)}"
            )
        for path in main + practice:
            if not path.is_file():
                errors.append(f"Missing stimulus: {path}")

    if errors:
        raise RuntimeError("Stimulus validation failed:\n" + "\n".join(errors))
    return counts


def _two_randomized_cycles(file_count: int, rng: random.Random) -> list[int]:
    """Match Presentation: show a full set before repeating any image."""
    order: list[int] = []
    while len(order) < TRIALS_PER_CONDITION:
        cycle = list(range(1, file_count + 1))
        rng.shuffle(cycle)
        order.extend(cycle)
    return order[:TRIALS_PER_CONDITION]


def build_main_trials(
    rng: random.Random,
    object_key: str,
    scrambled_key: str,
) -> list[Trial]:
    """Create eight shuffled blocks with 10 trials per condition per block."""
    files_by_condition = {condition: condition_files(condition) for condition in CONDITIONS}
    orders = {
        condition: _two_randomized_cycles(len(files_by_condition[condition]), rng)
        for condition in CONDITIONS
    }

    counters = {condition: 0 for condition in CONDITIONS}

    trials: list[Trial] = []
    sequence_number = 0
    for block_number in range(1, BLOCKS + 1):
        block_conditions = [
            condition
            for condition in CONDITIONS
            for _ in range(TRIALS_PER_CONDITION_PER_BLOCK)
        ]
        rng.shuffle(block_conditions)
        for condition in block_conditions:
            sequence_number += 1
            order_position = counters[condition]
            stimulus_index = orders[condition][order_position]
            counters[condition] += 1
            path = files_by_condition[condition][stimulus_index - 1]
            trials.append(
                Trial(
                    phase="Main",
                    block_number=block_number,
                    trial_number=sequence_number,
                    type_name=condition.type_name,
                    condition_name=condition.condition_name,
                    stimulus_path=path,
                    stimulus_index=stimulus_index,
                    port_code=stimulus_code(condition, stimulus_index),
                    correct_key=(
                        object_key
                        if condition.type_name == "Object"
                        else scrambled_key
                    ),
                )
            )
    return trials


def build_practice_trials(
    rng: random.Random,
    object_key: str,
    scrambled_key: str,
) -> list[Trial]:
    """Match the original flat practice arrays, truncation, and final shuffle."""
    by_type = {
        "Object": [
            next(c for c in CONDITIONS if c.condition_name == "Cars"),
            next(c for c in CONDITIONS if c.condition_name == "Faces"),
        ],
        "Scrambled": [
            next(c for c in CONDITIONS if c.condition_name == "Scrambled Cars"),
            next(c for c in CONDITIONS if c.condition_name == "Scrambled Faces"),
        ],
    }

    pool: list[Trial] = []
    for type_name in ("Object", "Scrambled"):
        type_index = 0
        for condition in by_type[type_name]:
            for path in practice_files(condition):
                type_index += 1
                # Presentation assigns all practice images condition index 1.
                code = type_index + (100 if type_name == "Scrambled" else 0)
                pool.append(
                    Trial(
                        phase="Practice",
                        block_number=0,
                        trial_number=0,
                        type_name=type_name,
                        condition_name="Practice",
                        stimulus_path=path,
                        stimulus_index=type_index,
                        port_code=code,
                        correct_key=object_key if type_name == "Object" else scrambled_key,
                    )
                )

    selected = pool[:PRACTICE_TRIALS]
    rng.shuffle(selected)
    return [
        Trial(**{**asdict(trial), "trial_number": number})
        for number, trial in enumerate(selected, start=1)
    ]


def validate_design(seed: int = 170) -> dict[str, object]:
    counts = validate_stimuli()
    rng = random.Random(seed)
    practice = build_practice_trials(rng, "up", "down")
    main = build_main_trials(rng, "up", "down")

    condition_counts = {
        condition.condition_name: sum(
            trial.condition_name == condition.condition_name for trial in main
        )
        for condition in CONDITIONS
    }
    code_ranges = {
        condition.condition_name: (
            min(trial.port_code for trial in main if trial.condition_name == condition.condition_name),
            max(trial.port_code for trial in main if trial.condition_name == condition.condition_name),
        )
        for condition in CONDITIONS
    }
    unique_repetitions = {
        condition.condition_name: sorted(
            {
                sum(
                    trial.stimulus_path == path
                    for trial in main
                    if trial.condition_name == condition.condition_name
                )
                for path in condition_files(condition)
            }
        )
        for condition in CONDITIONS
    }
    block_condition_counts = {
        str(block_number): {
            condition.condition_name: sum(
                trial.condition_name == condition.condition_name
                and trial.block_number == block_number
                for trial in main
            )
            for condition in CONDITIONS
        }
        for block_number in range(1, BLOCKS + 1)
    }

    report = {
        "seed": seed,
        "refresh_hz": EXPECTED_REFRESH_HZ,
        "stimulus_frames": frames_for_ms(STIMULUS_DURATION_MS),
        "stimulus_actual_ms": frames_for_ms(STIMULUS_DURATION_MS) / EXPECTED_REFRESH_HZ * 1000,
        "isi_frame_range": [frames_for_ms(ISI_MIN_MS), frames_for_ms(ISI_MAX_MS)],
        "ready_frames": frames_for_ms(READY_DURATION_MS),
        "minimum_break_duration_s": MINIMUM_BREAK_DURATION_S,
        "break_can_extend_until_ready": True,
        "display_defaults": {
            "screen_size_px": DEFAULT_SCREEN_SIZE_PX,
            "native_screen_size_px": NATIVE_SCREEN_SIZE_PX,
            "width_cm": DISPLAY_WIDTH_CM,
            "height_cm": DISPLAY_HEIGHT_CM,
            "viewing_distance_cm": VIEWING_DISTANCE_CM,
            "retina": True,
        },
        "stimulus_counts": counts,
        "practice_trials": len(practice),
        "main_trials": len(main),
        "blocks": BLOCKS,
        "trials_per_block": TRIALS_BETWEEN_BREAKS,
        "trials_per_condition_per_block": TRIALS_PER_CONDITION_PER_BLOCK,
        "condition_trial_counts": condition_counts,
        "block_condition_counts": block_condition_counts,
        "code_ranges": code_ranges,
        "repetitions_per_main_image": unique_repetitions,
    }

    if len(main) != 320 or len(practice) != 10:
        raise AssertionError("Unexpected trial count")
    if any(count != 80 for count in condition_counts.values()):
        raise AssertionError(f"Unbalanced condition counts: {condition_counts}")
    if any(
        count != TRIALS_PER_CONDITION_PER_BLOCK
        for block_counts in block_condition_counts.values()
        for count in block_counts.values()
    ):
        raise AssertionError(
            f"Unbalanced within-block condition counts: {block_condition_counts}"
        )
    if any(repetitions != [2] for repetitions in unique_repetitions.values()):
        raise AssertionError(f"Unexpected image repetitions: {unique_repetitions}")
    return report


def run_visual_smoke_test() -> int:
    """Open a short window and exercise the actual PsychoPy drawing path."""
    try:
        from psychopy import monitors, visual
    except ImportError as exc:
        raise RuntimeError("PsychoPy is required for the visual smoke test") from exc

    validate_stimuli()
    monitor = monitors.Monitor(
        "ERP_CORE_N170_smoke",
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
    try:
        image = visual.ImageStim(
            win,
            image=str(condition_files(CONDITIONS[0])[0]),
            size=(IMAGE_WIDTH_DEG, IMAGE_HEIGHT_DEG),
            units="deg",
            interpolate=True,
        )
        fixation = visual.Circle(
            win,
            radius=FIXATION_DIAMETER_DEG / 2.0,
            fillColor=FIXATION_RGB255,
            lineColor=FIXATION_RGB255,
            colorSpace="rgb255",
            edges=64,
        )
        for _ in range(frames_for_ms(STIMULUS_DURATION_MS)):
            image.draw()
            win.flip()
        for _ in range(12):
            fixation.draw()
            win.flip()
    finally:
        win.close()
    print("Visual smoke test completed.")
    return 0


def sanitize_bids_label(value: str, prefix: str, fallback: str) -> str:
    """Return an alphanumeric BIDS label without an entity prefix."""
    value = value.strip()
    value = re.sub(rf"^{re.escape(prefix)}-", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]+", "", value)
    return value or fallback


def ensure_bids_dataset_root(participant: str) -> None:
    """Create the minimal dataset-level BIDS metadata without overwriting it."""
    DATA_DIR.mkdir(exist_ok=True)
    description_path = DATA_DIR / "dataset_description.json"
    if not description_path.exists():
        description_path.write_text(
            json.dumps(
                {
                    "Name": "ERP CORE N170 PsychoPy",
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

    rows.append({name: participant_id if name == "participant_id" else "n/a" for name in fieldnames})
    with participants_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace, program_phase: str) -> int:
    if program_phase not in {"practice", "main"}:
        raise ValueError(f"Unknown N170 program phase: {program_phase}")

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
        title=f"ERP CORE N170 - {program_phase.title()}",
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
        output_stem = f"{subject_label}_task-n170_run-{run_id}"
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
    object_key = args.object_key
    scrambled_key = "down" if object_key == "up" else "up"
    seed = args.seed if args.seed is not None else int(time.time_ns() % (2**32))
    rng = random.Random(seed)

    validate_stimuli()
    if program_phase == "practice":
        selected_trials = build_practice_trials(rng, object_key, scrambled_key)
    else:
        selected_trials = build_main_trials(rng, object_key, scrambled_key)

    screen_size = (args.screen_width_px, args.screen_height_px)
    window_size = (args.window_width_px, args.window_height_px)
    display_width_cm = args.display_width_cm
    display_height_cm = args.display_height_cm
    viewing_distance_cm = args.viewing_distance_cm
    monitor = monitors.Monitor(
        "ERP_CORE_N170",
        width=display_width_cm,
        distance=viewing_distance_cm,
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
            if args.send_parallel_triggers:
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
    fixation = visual.Circle(
        win,
        radius=FIXATION_DIAMETER_DEG / 2.0,
        fillColor=FIXATION_RGB255,
        lineColor=FIXATION_RGB255,
        colorSpace="rgb255",
        edges=64,
    )

    image_paths = sorted(
        {trial.stimulus_path for trial in selected_trials},
        key=lambda path: (str(path.parent), natural_key(path)),
    )
    images = {
        path: visual.ImageStim(
            win,
            image=str(path),
            size=(IMAGE_WIDTH_DEG, IMAGE_HEIGHT_DEG),
            units="deg",
            interpolate=True,
        )
        for path in image_paths
    }

    object_name = f"the {object_key.upper()} arrow"
    scrambled_name = f"the {scrambled_key.upper()} arrow"
    instructions = (
        "Decide whether each image is an object (a face or car) or a texture.\n\n"
        f"Face/Car: press {object_name}.\n"
        f"Texture: press {scrambled_name}.\n\n"
        "Respond quickly and accurately."
    )
    reminder = (
        f"Face/Car: press {object_name}.\n"
        f"Texture: press {scrambled_name}.\n\n"
        "Press either button to begin."
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
            keyList=[object_key, scrambled_key, "escape"],
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
        frame_count = frames_for_ms(READY_DURATION_MS)
        for word in ("Ready", "Set", "Go"):
            show_timed_text(word, frame_count)

    def show_break_countdown() -> None:
        event.clearEvents(eventType="keyboard")
        frames_per_second = frames_for_ms(1000)
        text_stim.height = INSTRUCTION_TEXT_HEIGHT_DEG
        text_stim.alignText = "center"
        for seconds_remaining in range(MINIMUM_BREAK_DURATION_S, 0, -1):
            text_stim.text = (
                "Take a rest!\n\n"
                f"Minimum rest: {seconds_remaining} seconds remaining."
            )
            for _ in range(frames_per_second):
                text_stim.draw()
                win.flip()
                abort_if_requested(event.getKeys(keyList=["escape"]))
        event.clearEvents(eventType="keyboard")
        show_message(
            "The minimum break is complete.\n\n"
            "Continue resting if needed. Press either response button when "
            "you are ready to continue."
        )

    def show_fixation(frames: int) -> None:
        for _ in range(frames):
            fixation.draw()
            win.flip()
            abort_if_requested(event.getKeys(keyList=["escape"]))

    def collect_first_response(
        response: Optional[KeyResponse],
        valid_keys: list[str],
    ) -> Optional[KeyResponse]:
        keys = event.getKeys(keyList=valid_keys, timeStamped=response_clock)
        abort_if_requested(keys)
        if response is None:
            task_keys = [
                (name, rt)
                for name, rt in keys
                if name in (object_key, scrambled_key)
            ]
            if task_keys:
                name, rt = min(task_keys, key=lambda key: key[1])
                response = KeyResponse(name=name, rt=float(rt))
        return response

    def send_new_response_trigger(
        previous_response: Optional[KeyResponse],
        response: Optional[KeyResponse],
        correct_key: str,
    ) -> None:
        if previous_response is None and response is not None:
            response_code = (
                CORRECT_RESPONSE_CODE
                if response.name == correct_key
                else INCORRECT_RESPONSE_CODE
            )
            trigger.send(response_code)
            win.callOnFlip(trigger.reset)

    data_rows: list[dict[str, object]] = []
    experiment_status = "completed"
    experiment_clock = core.Clock()

    def mark_experiment_time(target: dict[str, float], key: str) -> None:
        target[key] = experiment_clock.getTime()

    def present_trial(trial: Trial) -> None:
        requested_isi_ms = rng.randint(ISI_MIN_MS, ISI_MAX_MS)
        isi_frames = frames_for_ms(requested_isi_ms)
        stimulus_frames = frames_for_ms(STIMULUS_DURATION_MS)
        response = None
        valid_keys = [object_key, scrambled_key, "escape"]

        # Record frame intervals only during trial presentation. Instruction
        # and rest screens wait for input and must not count as dropped frames.
        win.recordFrameIntervals = True
        flip_times: dict[str, float] = {}
        for frame in range(stimulus_frames):
            images[trial.stimulus_path].draw()
            if frame == 0:
                win.callOnFlip(response_clock.reset)
                win.callOnFlip(event.clearEvents, eventType="keyboard")
                win.callOnFlip(trigger.send, trial.port_code)
                win.callOnFlip(mark_experiment_time, flip_times, "onset")
            elif frame == 1:
                win.callOnFlip(trigger.reset)
            win.flip()
            previous_response = response
            response = collect_first_response(response, valid_keys)
            send_new_response_trigger(previous_response, response, trial.correct_key)

        for frame in range(isi_frames):
            fixation.draw()
            if frame == 0:
                win.callOnFlip(mark_experiment_time, flip_times, "offset")
            win.flip()
            previous_response = response
            response = collect_first_response(response, valid_keys)
            send_new_response_trigger(previous_response, response, trial.correct_key)
        win.recordFrameIntervals = False

        if response is not None and response.rt is not None:
            response_key = response.name
            response_rt_s = float(response.rt)
            accuracy = int(response_key == trial.correct_key)
            response_code = CORRECT_RESPONSE_CODE if accuracy else INCORRECT_RESPONSE_CODE
        else:
            response_key = ""
            response_rt_s = ""
            accuracy = 0
            response_code = ""

        data_rows.append(
            {
                "participant": participant,
                "run": run_id,
                "seed": seed,
                "phase": trial.phase,
                "block_number": trial.block_number,
                "trial_number": trial.trial_number,
                "stim_type": trial.type_name,
                "stim_condition": trial.condition_name,
                "stimulus_file": trial.stimulus_path.name,
                "stimulus_relative_path": str(trial.stimulus_path.relative_to(BASE_DIR)),
                "stimulus_index": trial.stimulus_index,
                "stimulus_code": trial.port_code,
                "correct_key": trial.correct_key,
                "response_key": response_key,
                "accuracy": accuracy,
                "rt_s": response_rt_s,
                "response_code": response_code,
                "requested_isi_ms": requested_isi_ms,
                "isi_frames": isi_frames,
                "frame_quantized_isi_ms": isi_frames / EXPECTED_REFRESH_HZ * 1000.0,
                "stimulus_frames": stimulus_frames,
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
            show_message(reminder)
            ready_set_go()
            show_fixation(frames_for_ms(rng.randint(ISI_MIN_MS, ISI_MAX_MS)))
            for index, trial in enumerate(selected_trials, start=1):
                present_trial(trial)
                if (
                    index % TRIALS_BETWEEN_BREAKS == 0
                    and index < len(selected_trials)
                ):
                    show_break_countdown()
                    ready_set_go()
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
        "stimulus_file",
        "value",
        "block_number",
        "trial_number",
        "stimulus_category",
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
        condition = str(row["stim_condition"])
        if condition == "Practice":
            trial_type = f"practice_{str(row['stim_type']).lower()}"
        else:
            trial_type = re.sub(r"[^a-z0-9]+", "_", condition.lower()).strip("_")
        presentation_onset = float(row["stimulus_onset_s"])
        measured_duration_ms = float(row["measured_stimulus_ms"])
        event_rows.append(
            {
                "onset": presentation_onset - first_onset_s,
                "duration": measured_duration_ms / 1000.0,
                "trial_type": trial_type,
                "response_time": row["rt_s"] if row["rt_s"] != "" else "n/a",
                "stimulus_file": row["stimulus_file"],
                "value": row["stimulus_code"],
                "block_number": row["block_number"],
                "trial_number": row["trial_number"],
                "stimulus_category": row["stim_type"],
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
                "Stimulus onset relative to the first stimulus in the PsychoPy "
                "log. Align to EEG acquisition time using stimulus trigger values."
            ),
            "Units": "s",
        },
        "duration": {
            "Description": "Measured time from stimulus-onset flip to fixation-onset flip.",
            "Units": "s",
        },
        "trial_type": {"Description": "Experimental condition."},
        "response_time": {
            "Description": "First response time relative to stimulus onset.",
            "Units": "s",
        },
        "stimulus_file": {"Description": "Original ERP CORE bitmap filename."},
        "value": {"Description": "Stimulus trigger code."},
        "block_number": {
            "Description": "Main-experiment block number; practice uses 0."
        },
        "trial_number": {"Description": "Trial number within this program."},
        "stimulus_category": {"Description": "Object or Scrambled category."},
        "correct_response": {"Description": "Correct arrow-key response."},
        "response": {"Description": "First recorded arrow-key response."},
        "accuracy": {"Description": "Response accuracy, coded 0 or 1."},
        "response_value": {"Description": "Response trigger code, 201 or 202."},
        "requested_isi_ms": {"Description": "Randomly requested ISI.", "Units": "ms"},
        "isi_frames": {"Description": "ISI duration in 60 Hz video frames."},
        "frame_quantized_isi_ms": {
            "Description": "ISI duration after conversion to whole 60 Hz frames.",
            "Units": "ms",
        },
        "stimulus_frames": {"Description": "Stimulus duration in 60 Hz video frames."},
        "presentation_onset_s": {
            "Description": "Raw PsychoPy experiment-clock timestamp at stimulus onset.",
            "Units": "s",
        },
        "presentation_offset_s": {
            "Description": "Raw PsychoPy experiment-clock timestamp at stimulus offset.",
            "Units": "s",
        },
        "measured_stimulus_ms": {
            "Description": "Stimulus duration measured from flip timestamps.",
            "Units": "ms",
        },
        "random_seed": {"Description": "Random seed used to construct this run."},
    }
    events_json_path.write_text(
        json.dumps(events_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    dropped_threshold_s = 1.5 / EXPECTED_REFRESH_HZ
    stimulus_durations_ms = [
        float(row["measured_stimulus_ms"])
        for row in data_rows
        if isinstance(row.get("measured_stimulus_ms"), (int, float))
    ]
    one_frame_ms = 1000.0 / EXPECTED_REFRESH_HZ
    stimulus_timing = {
        "target_ms": STIMULUS_DURATION_MS,
        "samples": len(stimulus_durations_ms),
        "mean_ms": (
            statistics.fmean(stimulus_durations_ms)
            if stimulus_durations_ms
            else None
        ),
        "median_ms": (
            statistics.median(stimulus_durations_ms)
            if stimulus_durations_ms
            else None
        ),
        "minimum_ms": min(stimulus_durations_ms) if stimulus_durations_ms else None,
        "maximum_ms": max(stimulus_durations_ms) if stimulus_durations_ms else None,
        "mean_absolute_error_ms": (
            statistics.fmean(
                abs(duration - STIMULUS_DURATION_MS)
                for duration in stimulus_durations_ms
            )
            if stimulus_durations_ms
            else None
        ),
        "within_one_60hz_frame": sum(
            abs(duration - STIMULUS_DURATION_MS) <= one_frame_ms
            for duration in stimulus_durations_ms
        ),
    }
    summary = {
        "participant": participant,
        "run": run_id,
        "program_phase": program_phase,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "events_file": str(events_path.relative_to(DATA_DIR)),
        "onset_reference": (
            "First stimulus in the PsychoPy log; align to EEG acquisition "
            "using stimulus trigger values."
        ),
        "seed": seed,
        "status": experiment_status,
        "error": repr(runtime_error) if runtime_error is not None else None,
        "assumed_refresh_hz": EXPECTED_REFRESH_HZ,
        "trigger_mode": trigger.mode,
        "object_key": object_key,
        "scrambled_key": scrambled_key,
        "completed_trials": len(data_rows),
        "planned_trials": len(selected_trials),
        "trials_between_breaks": TRIALS_BETWEEN_BREAKS,
        "minimum_break_duration_s": MINIMUM_BREAK_DURATION_S,
        "break_can_extend_until_ready": True,
        "elapsed_s": elapsed_s,
        "recorded_frame_intervals": len(frame_intervals),
        "dropped_frame_intervals": sum(
            interval > dropped_threshold_s for interval in frame_intervals
        ),
        "longest_frame_interval_ms": (
            max(frame_intervals) * 1000.0 if frame_intervals else None
        ),
        "stimulus_timing_from_flip_timestamps": stimulus_timing,
        "display": {
            "window_size_px": window_size,
            "screen_size_px": screen_size,
            "native_screen_size_px": NATIVE_SCREEN_SIZE_PX,
            "fullscreen": False,
            "retina": True,
            "width_cm": display_width_cm,
            "height_cm": display_height_cm,
            "viewing_distance_cm": viewing_distance_cm,
        },
    }
    timing_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved events: {events_path}")
    print(f"Saved events metadata: {events_json_path}")
    print(f"Saved timing diagnostics: {timing_path}")
    if stimulus_durations_ms:
        print(
            "Stimulus timing from flip timestamps: "
            f"mean={stimulus_timing['mean_ms']:.3f} ms, "
            f"min={stimulus_timing['minimum_ms']:.3f} ms, "
            f"max={stimulus_timing['maximum_ms']:.3f} ms"
        )
    if runtime_error is not None:
        raise runtime_error
    return 0 if experiment_status == "completed" else 130


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="check stimuli and trial construction without opening PsychoPy",
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
        "--object-key",
        choices=("up", "down"),
        default="up",
        help="arrow key assigned to intact objects; the other key means scrambled",
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
        help="display width reported to PsychoPy; use scaled resolution on Retina macOS",
    )
    parser.add_argument(
        "--screen-height-px",
        type=int,
        default=DEFAULT_SCREEN_SIZE_PX[1],
        help="display height reported to PsychoPy; use scaled resolution on Retina macOS",
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
        help="physical display height recorded in the output summary",
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
        help="send EEG event codes through a parallel port",
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
            "Choose one program: n170_practice.py for practice or "
            "n170_main.py for the main experiment."
        )
        return 2
    return run_experiment(args, program_phase)


if __name__ == "__main__":
    sys.exit(main())
