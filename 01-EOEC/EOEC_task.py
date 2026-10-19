#!/usr/bin/env python3
"""Alternating eyes-open / eyes-closed resting-state EEG experiment.

The task uses a fixed 60 Hz timing model and never measures refresh rate.
Eight 60-second blocks alternate between eyes open and eyes closed. High- and
low-tone meanings are counterbalanced across participants. There is no
transition interval: the tone, condition trigger, and block onset share the
same flip.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Fixed experiment configuration
# ---------------------------------------------------------------------------

EXPECTED_REFRESH_HZ = 60.0
DEFAULT_SCREEN_SIZE_PX = (1470, 956)
DEFAULT_WINDOW_SIZE_PX = (1100, 700)

BACKGROUND_RGB255 = (100, 100, 100)
TEXT_RGB255 = (0, 0, 0)
INSTRUCTION_TEXT_HEIGHT_PX = 22
# Equivalent to 0.72 degrees with the configured 29 cm-wide, 1470 px display
# viewed from 40 cm, matching the other experiments' Ready/Set/Go text.
READY_TEXT_HEIGHT_PX = 25.5

BLOCK_DURATION_S = 60.0
BLOCKS_PER_CONDITION = 4
TOTAL_BLOCKS = BLOCKS_PER_CONDITION * 2
ANALYSIS_DISCARD_INITIAL_S = 1.0
READY_DURATION_S = 1.0

HIGH_TONE_FREQUENCY_HZ = 1000.0
LOW_TONE_FREQUENCY_HZ = 400.0
TONE_DURATION_S = 0.3
TONE_VOLUME = 0.5

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class Condition:
    eye_state: str
    label: str
    trigger_code: int


EYES_OPEN = Condition(
    eye_state="eyes_open",
    label="EYES OPEN",
    trigger_code=11,
)
EYES_CLOSED = Condition(
    eye_state="eyes_closed",
    label="EYES CLOSED",
    trigger_code=12,
)
CONDITIONS = (EYES_OPEN, EYES_CLOSED)


def frames_for_seconds(
    duration_s: float,
    refresh_hz: float = EXPECTED_REFRESH_HZ,
) -> int:
    """Quantize a duration to the nearest whole video frame."""
    return max(1, int(round(duration_s * refresh_hz)))


def sanitize_bids_label(value: str, prefix: str, fallback: str) -> str:
    """Return an alphanumeric BIDS label without an entity prefix."""
    value = value.strip()
    value = re.sub(rf"^{re.escape(prefix)}-", "", value, flags=re.IGNORECASE)
    value = re.sub(r"[^A-Za-z0-9]+", "", value)
    return value or fallback


def counterbalance_group_for_participant(participant: str) -> int:
    """Assign one stable cell of the 2 x 2 counterbalancing design."""
    if participant.isdigit() and int(participant) > 0:
        return ((int(participant) - 1) % 4) + 1
    digest = hashlib.sha256(participant.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % 4) + 1


def start_condition_for_group(group: int) -> Condition:
    if not 1 <= group <= 4:
        raise ValueError("Counterbalance group must be 1-4")
    return EYES_OPEN if group in {1, 3} else EYES_CLOSED


def tone_frequencies_for_group(group: int) -> dict[Condition, float]:
    """Return the open/closed tone mapping for a counterbalance group."""
    if not 1 <= group <= 4:
        raise ValueError("Counterbalance group must be 1-4")
    if group in {1, 2}:
        return {
            EYES_OPEN: HIGH_TONE_FREQUENCY_HZ,
            EYES_CLOSED: LOW_TONE_FREQUENCY_HZ,
        }
    return {
        EYES_OPEN: LOW_TONE_FREQUENCY_HZ,
        EYES_CLOSED: HIGH_TONE_FREQUENCY_HZ,
    }


def start_condition_for_participant(participant: str) -> Condition:
    return start_condition_for_group(counterbalance_group_for_participant(participant))


def tone_frequencies_for_participant(participant: str) -> dict[Condition, float]:
    return tone_frequencies_for_group(counterbalance_group_for_participant(participant))


def build_sequence(participant: str) -> list[Condition]:
    """Build eight strictly alternating, participant-counterbalanced blocks."""
    first = start_condition_for_participant(participant)
    second = EYES_CLOSED if first == EYES_OPEN else EYES_OPEN
    return [first if index % 2 == 0 else second for index in range(TOTAL_BLOCKS)]


def validate_design(participant: str = "001") -> dict[str, object]:
    sequence = build_sequence(participant)
    counterbalance_group = counterbalance_group_for_participant(participant)
    tone_frequencies = tone_frequencies_for_group(counterbalance_group)
    counts = {
        condition.eye_state: sequence.count(condition) for condition in CONDITIONS
    }
    block_frames = frames_for_seconds(BLOCK_DURATION_S)
    report = {
        "participant": participant,
        "refresh_hz": EXPECTED_REFRESH_HZ,
        "refresh_rate_measured": False,
        "block_duration_s": BLOCK_DURATION_S,
        "block_frames": block_frames,
        "total_blocks": len(sequence),
        "blocks_per_condition": counts,
        "counterbalance_group": counterbalance_group,
        "start_condition": sequence[0].eye_state,
        "sequence": [condition.eye_state for condition in sequence],
        "tone_frequencies_hz": {
            condition.eye_state: tone_frequencies[condition]
            for condition in CONDITIONS
        },
        "tone_duration_s": TONE_DURATION_S,
        "trigger_codes": {
            condition.eye_state: condition.trigger_code for condition in CONDITIONS
        },
        "background_rgb255": BACKGROUND_RGB255,
        "text_rgb255": TEXT_RGB255,
        "ready_set_go_rgb255": TEXT_RGB255,
        "fixation_present": False,
        "transition_interval_s": 0.0,
        "analysis_discard_initial_s": ANALYSIS_DISCARD_INITIAL_S,
        "ready_set_go_duration_s_each": READY_DURATION_S,
    }
    if len(sequence) != TOTAL_BLOCKS:
        raise AssertionError("Unexpected number of blocks")
    if set(counts.values()) != {BLOCKS_PER_CONDITION}:
        raise AssertionError(f"Unbalanced conditions: {counts}")
    if any(left == right for left, right in zip(sequence, sequence[1:])):
        raise AssertionError("The sequence is not strictly alternating")
    if block_frames != 3600:
        raise AssertionError("A 60-second block at 60 Hz must contain 3600 frames")
    if len({condition.trigger_code for condition in CONDITIONS}) != 2:
        raise AssertionError("Condition trigger codes must be unique")
    return report


def ensure_bids_dataset_root(participant: str, counterbalance_group: int) -> None:
    """Create minimal dataset metadata and update the participant table."""
    DATA_DIR.mkdir(exist_ok=True)
    description_path = DATA_DIR / "dataset_description.json"
    if not description_path.exists():
        description_path.write_text(
            json.dumps(
                {
                    "Name": "Eyes-Open Eyes-Closed Resting-State EEG",
                    "BIDSVersion": "1.11.2",
                    "DatasetType": "raw",
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
        if "eoec_counterbalance_group" not in fieldnames:
            fieldnames.append("eoec_counterbalance_group")
            for row in rows:
                row["eoec_counterbalance_group"] = "n/a"
    else:
        fieldnames = [
            "participant_id",
            "eoec_counterbalance_group",
        ]
        rows = []

    participant_row = next(
        (row for row in rows if row.get("participant_id") == participant_id),
        None,
    )
    if participant_row is None:
        participant_row = {name: "n/a" for name in fieldnames}
        participant_row["participant_id"] = participant_id
        rows.append(participant_row)
    participant_row["eoec_counterbalance_group"] = str(counterbalance_group)

    with participants_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run_smoke_test() -> int:
    """Open a brief window and play the two tones."""
    try:
        from psychopy import core, sound, visual
    except ImportError as exc:
        raise RuntimeError("PsychoPy is required for the smoke test") from exc

    win = visual.Window(
        size=(800, 600),
        fullscr=False,
        units="pix",
        color=BACKGROUND_RGB255,
        colorSpace="rgb255",
        waitBlanking=True,
        checkTiming=False,
        allowGUI=True,
        useRetina=True,
    )
    message = visual.TextStim(
        win,
        text="High tone: eyes open",
        color=TEXT_RGB255,
        colorSpace="rgb255",
        height=28,
    )
    high_tone = sound.Sound(
        value=HIGH_TONE_FREQUENCY_HZ,
        secs=TONE_DURATION_S,
        volume=TONE_VOLUME,
        hamming=True,
    )
    low_tone = sound.Sound(
        value=LOW_TONE_FREQUENCY_HZ,
        secs=TONE_DURATION_S,
        volume=TONE_VOLUME,
        hamming=True,
    )
    try:
        message.draw()
        win.callOnFlip(high_tone.play)
        win.flip()
        core.wait(0.8)
        message.text = "Low tone: eyes closed"
        message.draw()
        win.callOnFlip(low_tone.play)
        win.flip()
        core.wait(0.8)
    finally:
        win.close()
    print("Eyes-open / eyes-closed smoke test completed.")
    return 0


def run_experiment(args: argparse.Namespace) -> int:
    try:
        from psychopy import core, event, gui, logging, sound, visual
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
        title="Eyes Open / Eyes Closed",
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

    sequence = build_sequence(participant)
    counterbalance_group = counterbalance_group_for_participant(participant)
    tone_frequencies = tone_frequencies_for_group(counterbalance_group)
    print(
        f"Participant {participant}: counterbalance group {counterbalance_group}; "
        f"first={sequence[0].eye_state}; "
        f"eyes_open={tone_frequencies[EYES_OPEN]:g} Hz; "
        f"eyes_closed={tone_frequencies[EYES_CLOSED]:g} Hz"
    )
    subject_label = f"sub-{participant}"
    output_dir = DATA_DIR / subject_label / "eeg"
    output_stem = f"{subject_label}_task-eoec_run-{run_id}"
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

    window_size = (args.window_width_px, args.window_height_px)
    screen_size = (args.screen_width_px, args.screen_height_px)
    win = visual.Window(
        size=window_size,
        fullscr=False,
        units="pix",
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

    high_tone = sound.Sound(
        value=HIGH_TONE_FREQUENCY_HZ,
        secs=TONE_DURATION_S,
        volume=TONE_VOLUME,
        hamming=True,
    )
    low_tone = sound.Sound(
        value=LOW_TONE_FREQUENCY_HZ,
        secs=TONE_DURATION_S,
        volume=TONE_VOLUME,
        hamming=True,
    )
    sound_by_frequency = {
        HIGH_TONE_FREQUENCY_HZ: high_tone,
        LOW_TONE_FREQUENCY_HZ: low_tone,
    }
    tone_by_condition = {
        condition: sound_by_frequency[tone_frequencies[condition]]
        for condition in CONDITIONS
    }
    tone_label_by_condition = {
        condition: (
            "HIGH" if tone_frequencies[condition] == HIGH_TONE_FREQUENCY_HZ else "LOW"
        )
        for condition in CONDITIONS
    }

    text_stim = visual.TextStim(
        win,
        text="",
        color=TEXT_RGB255,
        colorSpace="rgb255",
        font="Geneva",
        height=INSTRUCTION_TEXT_HEIGHT_PX,
        wrapWidth=900,
        alignText="left",
        anchorHoriz="center",
        anchorVert="center",
    )

    instructions = (
        "The experiment alternates between eyes-open and eyes-closed periods.\n\n"
        f"{tone_label_by_condition[EYES_OPEN]} tone: open your eyes and look "
        "toward the center of the blank screen.\n"
        f"{tone_label_by_condition[EYES_CLOSED]} tone: close your eyes.\n\n"
        "Stay relaxed and still. Each period lasts 60 seconds, and the next "
        "tone starts the next period. No response is required.\n\n"
        "Press SPACE to hear the tone demonstration."
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
        text_stim.height = INSTRUCTION_TEXT_HEIGHT_PX
        text_stim.alignText = align_text
        text_stim.draw()
        win.flip()
        keys = event.waitKeys(keyList=["space", "escape"], clearEvents=True)
        abort_if_requested(keys or [])
        event.clearEvents(eventType="keyboard")

    def show_tone_demo(label: str, tone: object) -> None:
        event.clearEvents(eventType="keyboard")
        text_stim.text = (
            f"{label}\n\n"
            "Press R to replay this tone.\n"
            "Press SPACE to continue."
        )
        text_stim.alignText = "center"
        text_stim.draw()
        win.callOnFlip(tone.play)
        win.flip()
        while True:
            keys = event.waitKeys(
                keyList=["r", "space", "escape"],
                clearEvents=True,
            )
            abort_if_requested(keys or [])
            if any(key_name(key) == "space" for key in keys or []):
                break
            tone.stop()
            tone.play()
        event.clearEvents(eventType="keyboard")

    def show_final_tone_demo(label: str, tone: object, first_condition: str) -> None:
        event.clearEvents(eventType="keyboard")
        text_stim.text = (
            f"{label}\n\n"
            "Press R to replay this tone.\n\n"
            f"The first period will be {first_condition}.\n"
            "Press SPACE when you are ready to begin the recording."
        )
        text_stim.alignText = "center"
        text_stim.draw()
        win.callOnFlip(tone.play)
        win.flip()
        while True:
            keys = event.waitKeys(
                keyList=["r", "space", "escape"],
                clearEvents=True,
            )
            abort_if_requested(keys or [])
            if any(key_name(key) == "space" for key in keys or []):
                break
            tone.stop()
            tone.play()
        event.clearEvents(eventType="keyboard")

    def show_timed_text(message: str, frames: int) -> None:
        text_stim.text = message
        text_stim.height = READY_TEXT_HEIGHT_PX
        text_stim.alignText = "center"
        for _ in range(frames):
            text_stim.draw()
            win.flip()
            check_escape()

    def ready_set_go() -> None:
        frames = frames_for_seconds(READY_DURATION_S)
        for word in ("Ready", "Set", "Go"):
            show_timed_text(word, frames)

    block_frames = frames_for_seconds(BLOCK_DURATION_S)
    total_frames = block_frames * len(sequence)
    experiment_clock = core.Clock()
    block_onsets: list[float] = []
    final_offset_s: Optional[float] = None
    experiment_status = "completed"
    runtime_error: Optional[Exception] = None

    def mark_block_onset(block_number: int, condition: Condition) -> None:
        block_onsets.append(experiment_clock.getTime())
        print(
            f"Block {block_number}/{len(sequence)} started: "
            f"{condition.eye_state} ({tone_frequencies[condition]:g} Hz tone)"
        )

    def mark_final_offset() -> None:
        nonlocal final_offset_s
        final_offset_s = experiment_clock.getTime()

    try:
        show_message(instructions)
        show_tone_demo(
            f"{tone_label_by_condition[EYES_OPEN]} TONE = EYES OPEN\n\n"
            "Keep your eyes open during this demonstration.",
            tone_by_condition[EYES_OPEN],
        )
        start_label = "eyes open" if sequence[0] == EYES_OPEN else "eyes closed"
        show_final_tone_demo(
            f"{tone_label_by_condition[EYES_CLOSED]} TONE = EYES CLOSED\n\n"
            "Keep your eyes open during this demonstration.",
            tone_by_condition[EYES_CLOSED],
            start_label,
        )
        ready_set_go()

        event.clearEvents(eventType="keyboard")
        experiment_clock.reset()
        win.frameIntervals.clear()
        win.recordFrameIntervals = True

        for global_frame in range(total_frames):
            within_block_frame = global_frame % block_frames
            if within_block_frame == 0:
                block_index = global_frame // block_frames
                condition = sequence[block_index]
                win.callOnFlip(tone_by_condition[condition].play)
                win.callOnFlip(trigger.send, condition.trigger_code)
                win.callOnFlip(mark_block_onset, block_index + 1, condition)
            elif within_block_frame == 1:
                win.callOnFlip(trigger.reset)

            # Drawing nothing leaves a uniform gray screen.
            win.flip()
            if global_frame == 0:
                # Do not count the long wait on the ready screen as a dropped
                # frame in the recording-period diagnostics.
                win.frameIntervals.clear()
            check_escape()

        win.recordFrameIntervals = False
        win.callOnFlip(mark_final_offset)
        if sequence[-1] == EYES_CLOSED:
            # Open the participant's eyes before displaying the completion page.
            win.callOnFlip(tone_by_condition[EYES_OPEN].play)
        text_stim.text = "Recording complete.\n\nThank you for participating."
        text_stim.alignText = "center"
        text_stim.draw()
        win.flip()
        event.clearEvents(eventType="keyboard")
        event.waitKeys(keyList=["space", "escape"], clearEvents=True)
    except KeyboardInterrupt:
        experiment_status = "interrupted"
        print("Experiment interrupted. Saving completed block data.")
    except Exception as exc:
        experiment_status = "error"
        runtime_error = exc
        print(f"Experiment error: {exc}. Saving completed block data.")
    finally:
        win.recordFrameIntervals = False
        elapsed_s = experiment_clock.getTime()
        frame_intervals = list(win.frameIntervals)
        trigger.close()
        win.close()

    # A block is complete only when the next boundary (or final boundary) was
    # displayed. This safely omits the active partial block after an abort.
    block_offsets = block_onsets[1:]
    if final_offset_s is not None:
        block_offsets.append(final_offset_s)
    completed_blocks = min(len(block_offsets), len(block_onsets), len(sequence))

    ensure_bids_dataset_root(participant, counterbalance_group)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_dir.mkdir(parents=True, exist_ok=True)

    first_onset_s = block_onsets[0] if block_onsets else 0.0
    event_fieldnames = [
        "onset",
        "duration",
        "trial_type",
        "value",
        "block_number",
        "eye_state",
        "beep_frequency_hz",
        "beep_duration_s",
        "analysis_onset",
        "analysis_duration",
        "discarded_initial_s",
        "refresh_rate_hz",
        "planned_frames",
        "presentation_onset_s",
        "presentation_offset_s",
        "measured_duration_ms",
        "counterbalance_group",
        "start_condition",
    ]
    event_rows: list[dict[str, object]] = []
    measured_durations_ms: list[float] = []
    for block_index in range(completed_blocks):
        condition = sequence[block_index]
        onset_s = block_onsets[block_index]
        offset_s = block_offsets[block_index]
        duration_s = offset_s - onset_s
        measured_durations_ms.append(duration_s * 1000.0)
        relative_onset_s = onset_s - first_onset_s
        event_rows.append(
            {
                "onset": relative_onset_s,
                "duration": duration_s,
                "trial_type": condition.eye_state,
                "value": condition.trigger_code,
                "block_number": block_index + 1,
                "eye_state": condition.eye_state,
                "beep_frequency_hz": tone_frequencies[condition],
                "beep_duration_s": TONE_DURATION_S,
                "analysis_onset": relative_onset_s + ANALYSIS_DISCARD_INITIAL_S,
                "analysis_duration": max(
                    0.0, duration_s - ANALYSIS_DISCARD_INITIAL_S
                ),
                "discarded_initial_s": ANALYSIS_DISCARD_INITIAL_S,
                "refresh_rate_hz": EXPECTED_REFRESH_HZ,
                "planned_frames": block_frames,
                "presentation_onset_s": onset_s,
                "presentation_offset_s": offset_s,
                "measured_duration_ms": duration_s * 1000.0,
                "counterbalance_group": counterbalance_group,
                "start_condition": sequence[0].eye_state,
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
            "Description": "Block onset relative to the first block onset",
            "Units": "s",
        },
        "duration": {
            "Description": "Measured duration between boundary flip timestamps",
            "Units": "s",
        },
        "trial_type": {
            "Description": "Resting-state eye condition",
            "Levels": {
                "eyes_open": "Eyes open while viewing a blank gray screen",
                "eyes_closed": "Eyes closed",
            },
        },
        "value": {
            "Description": "EEG trigger sent on the block-onset flip",
            "Levels": {"11": "Eyes open", "12": "Eyes closed"},
        },
        "beep_frequency_hz": {
            "Description": "Auditory instruction tone frequency",
            "Units": "Hz",
        },
        "analysis_onset": {
            "Description": "Suggested analysis onset after discarding the first second of the block",
            "Units": "s",
        },
        "analysis_duration": {
            "Description": "Suggested analyzed duration after the initial one-second exclusion",
            "Units": "s",
        },
        "discarded_initial_s": {
            "Description": "Recommended initial portion excluded during analysis, not during recording",
            "Units": "s",
        },
        "counterbalance_group": {
            "Description": "Participant-level 2 x 2 counterbalancing group",
            "Levels": {
                "1": "Eyes open first; high=open and low=closed",
                "2": "Eyes closed first; high=open and low=closed",
                "3": "Eyes open first; low=open and high=closed",
                "4": "Eyes closed first; low=open and high=closed",
            },
        },
    }
    events_json_path.write_text(
        json.dumps(events_metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    expected_interval_s = 1.0 / EXPECTED_REFRESH_HZ
    dropped_threshold_s = 1.5 * expected_interval_s

    def timing_stats(values: list[float], expected_ms: float) -> dict[str, object]:
        if not values:
            return {
                "n": 0,
                "expected_ms": expected_ms,
                "mean_ms": None,
                "minimum_ms": None,
                "maximum_ms": None,
                "mean_absolute_error_ms": None,
            }
        return {
            "n": len(values),
            "expected_ms": expected_ms,
            "mean_ms": statistics.mean(values),
            "minimum_ms": min(values),
            "maximum_ms": max(values),
            "mean_absolute_error_ms": statistics.mean(
                abs(value - expected_ms) for value in values
            ),
        }

    timing_summary = {
        "participant": participant,
        "run": run_id,
        "task": "eoec",
        "status": experiment_status,
        "completed_blocks": completed_blocks,
        "planned_blocks": len(sequence),
        "elapsed_s": elapsed_s,
        "expected_refresh_hz": EXPECTED_REFRESH_HZ,
        "refresh_rate_measured": False,
        "trigger_mode": trigger.mode,
        "display": {
            "screen_size_px": list(screen_size),
            "window_size_px": list(window_size),
            "fullscreen": False,
            "background_rgb255": list(BACKGROUND_RGB255),
            "fixation_present": False,
        },
        "paradigm": {
            "block_duration_s": BLOCK_DURATION_S,
            "block_frames": block_frames,
            "total_blocks": len(sequence),
            "blocks_per_condition": BLOCKS_PER_CONDITION,
            "transition_interval_s": 0.0,
            "counterbalance_group": counterbalance_group,
            "start_condition": sequence[0].eye_state,
            "sequence": [condition.eye_state for condition in sequence],
            "tone_frequencies_hz": {
                condition.eye_state: tone_frequencies[condition]
                for condition in CONDITIONS
            },
            "tone_duration_s": TONE_DURATION_S,
            "trigger_codes": {
                condition.eye_state: condition.trigger_code
                for condition in CONDITIONS
            },
            "recommended_analysis_discard_initial_s": ANALYSIS_DISCARD_INITIAL_S,
        },
        "timing_from_flip_timestamps": {
            "block": timing_stats(
                measured_durations_ms,
                BLOCK_DURATION_S * 1000.0,
            )
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
            "The beep request, trigger call, and block timestamp are scheduled "
            "on the same PsychoPy flip. Audio hardware and EEG interfaces may "
            "add latency. Verify end-to-end timing with appropriate hardware."
        ),
    }
    timing_path.write_text(
        json.dumps(timing_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved events: {events_path}")
    print(f"Saved events metadata: {events_json_path}")
    print(f"Saved timing diagnostics: {timing_path}")
    if measured_durations_ms:
        print(
            "Block timing from flip timestamps: "
            f"mean={statistics.mean(measured_durations_ms):.3f} ms, "
            f"min={min(measured_durations_ms):.3f} ms, "
            f"max={max(measured_durations_ms):.3f} ms"
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
        help="briefly open a PsychoPy window and play both tones",
    )
    parser.add_argument("--participant", default="debug", help="participant identifier")
    parser.add_argument("--run", default="01", help="numeric BIDS run label")
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
        "--send-parallel-triggers",
        action="store_true",
        help="send one-frame parallel-port block-onset triggers",
    )
    parser.add_argument(
        "--parallel-address",
        default="0x0378",
        help="parallel-port address parsed with Python base-0 syntax",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.validate_only:
        participant = sanitize_bids_label(
            args.participant,
            prefix="sub",
            fallback="anonymous",
        )
        print(json.dumps(validate_design(participant), indent=2))
        return 0
    if args.smoke_test:
        return run_smoke_test()
    return run_experiment(args)


if __name__ == "__main__":
    sys.exit(main())
