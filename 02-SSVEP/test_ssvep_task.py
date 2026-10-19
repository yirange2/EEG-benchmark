import csv
import random
import statistics
import tempfile
import unittest
from pathlib import Path

import ssvep_task


class SSVEPDesignTests(unittest.TestCase):
    def test_bids_labels_remove_prefixes_and_invalid_characters(self):
        self.assertEqual(
            ssvep_task.sanitize_bids_label("sub-P 001", "sub", "anonymous"),
            "P001",
        )
        self.assertEqual(
            ssvep_task.sanitize_bids_label("run-01", "run", "01"),
            "01",
        )

    def test_design_matches_agreed_parameters(self):
        report = ssvep_task.validate_design(seed=170)
        self.assertEqual(ssvep_task.READY_TEXT_HEIGHT_DEG, 0.72)
        self.assertEqual(report["practice_trials"], 4)
        self.assertEqual(report["main_trials"], 32)
        self.assertEqual(set(report["main_trials_per_target"].values()), {8})
        self.assertEqual(
            report["timing_frames"],
            {"cue": 30, "gaze_shift": 30, "stimulation": 420, "iti": 240},
        )
        self.assertEqual(report["stimulus_size_deg"], 3.6)
        self.assertEqual(
            report["attention_response_keys"],
            ("up", "right", "down", "left"),
        )
        self.assertEqual(
            report["attention_prompt_timing"],
            "immediately_after_stimulation",
        )
        self.assertTrue(report["iti_after_attention_response"])
        self.assertEqual(
            report["stimulus_positions_deg"],
            {
                "up": (0.0, 6.5),
                "right": (6.5, 0.0),
                "down": (0.0, -6.5),
                "left": (-6.5, 0.0),
            },
        )

    def test_frequency_location_mapping_and_trigger_codes(self):
        report = ssvep_task.validate_design(seed=170, participant="001")
        self.assertEqual(report["counterbalance_group"], 1)
        self.assertEqual(
            report["frequency_location_mapping"],
            {
                "up": 12.0,
                "right": 15.0,
                "down": 20.0,
                "left": 0.0,
            },
        )
        self.assertEqual(
            report["trigger_codes"],
            {"12 Hz": 12, "15 Hz": 15, "20 Hz": 20, "Static": 1},
        )

    def test_participant_level_latin_square_counterbalancing(self):
        self.assertEqual(
            [
                ssvep_task.counterbalance_group_for_participant(participant)
                for participant in ("001", "002", "003", "004", "005")
            ],
            [1, 2, 3, 4, 1],
        )

        mappings = [
            {
                target.location: target.frequency_hz
                for target in ssvep_task.targets_for_counterbalance_group(group)
            }
            for group in range(1, 5)
        ]
        for location in ssvep_task.LOCATION_ORDER:
            self.assertEqual(
                {mapping[location] for mapping in mappings},
                {0.0, 12.0, 15.0, 20.0},
            )

        first_assignment = ssvep_task.targets_for_participant("Alice")
        self.assertEqual(
            first_assignment,
            ssvep_task.targets_for_participant("Alice"),
        )

    def test_participants_tsv_records_counterbalance_group(self):
        original_data_dir = ssvep_task.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                ssvep_task.DATA_DIR = Path(temporary_directory) / "data"
                ssvep_task.ensure_bids_dataset_root("002", 2)
                participants_path = ssvep_task.DATA_DIR / "participants.tsv"
                with participants_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream, delimiter="\t"))
                self.assertEqual(
                    rows,
                    [
                        {
                            "participant_id": "sub-002",
                            "ssvep_counterbalance_group": "2",
                        }
                    ],
                )
        finally:
            ssvep_task.DATA_DIR = original_data_dir

    def test_flickering_frequencies_have_equal_temporal_mean_luminance(self):
        means = []
        for target in ssvep_task.TARGETS:
            if target.frequency_hz == 0:
                continue
            cycle_frames = round(
                ssvep_task.EXPECTED_REFRESH_HZ / target.frequency_hz
            )
            values = [
                ssvep_task.sine_luminance_rgb255(target.frequency_hz, frame)
                for frame in range(cycle_frames)
            ]
            means.append(statistics.mean(values))
        self.assertAlmostEqual(max(means), min(means), places=9)

    def test_static_circle_is_white_and_flicker_trough_matches_background(self):
        self.assertEqual(
            ssvep_task.sine_luminance_rgb255(0, 0),
            ssvep_task.STATIC_LUMINANCE_RGB255,
        )
        self.assertEqual(ssvep_task.STATIC_LUMINANCE_RGB255, 255.0)
        self.assertAlmostEqual(
            ssvep_task.sine_luminance_rgb255(15.0, 2),
            ssvep_task.BACKGROUND_LUMINANCE_RGB255,
        )

    def test_main_trials_are_balanced(self):
        trials = ssvep_task.build_trials(random.Random(170), "main")
        for target in ssvep_task.TARGETS:
            self.assertEqual(sum(trial.target == target for trial in trials), 8)


if __name__ == "__main__":
    unittest.main()
