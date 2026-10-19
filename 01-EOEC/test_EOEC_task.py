import csv
import tempfile
import unittest
from pathlib import Path

import EOEC_task as task


class EyesOpenClosedDesignTests(unittest.TestCase):
    def test_bids_labels_remove_prefixes_and_invalid_characters(self):
        self.assertEqual(
            task.sanitize_bids_label("sub-P 001", "sub", "anonymous"),
            "P001",
        )
        self.assertEqual(task.sanitize_bids_label("run-01", "run", "01"), "01")

    def test_60_second_block_has_3600_frames(self):
        self.assertEqual(task.frames_for_seconds(task.BLOCK_DURATION_S), 3600)

    def test_numeric_participants_cycle_through_four_groups(self):
        self.assertEqual(
            [
                task.counterbalance_group_for_participant(participant)
                for participant in ("001", "002", "003", "004", "005")
            ],
            [1, 2, 3, 4, 1],
        )
        self.assertEqual(task.start_condition_for_participant("001"), task.EYES_OPEN)
        self.assertEqual(task.start_condition_for_participant("002"), task.EYES_CLOSED)
        self.assertEqual(task.start_condition_for_participant("003"), task.EYES_OPEN)
        self.assertEqual(task.start_condition_for_participant("004"), task.EYES_CLOSED)

    def test_tone_mapping_is_balanced_across_groups(self):
        expected = {
            1: (1000.0, 400.0),
            2: (1000.0, 400.0),
            3: (400.0, 1000.0),
            4: (400.0, 1000.0),
        }
        for group, (open_hz, closed_hz) in expected.items():
            mapping = task.tone_frequencies_for_group(group)
            self.assertEqual(mapping[task.EYES_OPEN], open_hz)
            self.assertEqual(mapping[task.EYES_CLOSED], closed_hz)

    def test_sequence_is_balanced_and_strictly_alternating(self):
        for participant in ("001", "002", "Alice"):
            sequence = task.build_sequence(participant)
            self.assertEqual(len(sequence), 8)
            self.assertEqual(sequence.count(task.EYES_OPEN), 4)
            self.assertEqual(sequence.count(task.EYES_CLOSED), 4)
            self.assertTrue(
                all(left != right for left, right in zip(sequence, sequence[1:]))
            )

    def test_nonnumeric_assignment_is_stable(self):
        self.assertEqual(
            task.counterbalance_group_for_participant("Alice"),
            task.counterbalance_group_for_participant("Alice"),
        )

    def test_validate_design_records_no_transition_interval(self):
        self.assertEqual(task.READY_TEXT_HEIGHT_PX, 25.5)
        report = task.validate_design("001")
        self.assertEqual(report["transition_interval_s"], 0.0)
        self.assertEqual(report["analysis_discard_initial_s"], 1.0)
        self.assertEqual(report["ready_set_go_duration_s_each"], 1.0)
        self.assertEqual(report["text_rgb255"], (0, 0, 0))
        self.assertEqual(report["ready_set_go_rgb255"], (0, 0, 0))
        self.assertFalse(report["refresh_rate_measured"])
        self.assertFalse(report["fixation_present"])
        self.assertEqual(
            report["trigger_codes"],
            {"eyes_open": 11, "eyes_closed": 12},
        )

    def test_participants_tsv_records_counterbalance_group(self):
        original_data_dir = task.DATA_DIR
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                task.DATA_DIR = Path(temporary_directory) / "data"
                task.ensure_bids_dataset_root("002", 2)
                participants_path = task.DATA_DIR / "participants.tsv"
                with participants_path.open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.DictReader(stream, delimiter="\t"))
                self.assertEqual(
                    rows,
                    [
                        {
                            "participant_id": "sub-002",
                            "eoec_counterbalance_group": "2",
                        }
                    ],
                )
        finally:
            task.DATA_DIR = original_data_dir


if __name__ == "__main__":
    unittest.main()
