import random
import unittest

import n170_task


class N170DesignTests(unittest.TestCase):
    def test_bids_labels_remove_prefixes_and_invalid_characters(self):
        self.assertEqual(
            n170_task.sanitize_bids_label("sub-P 001", "sub", "anonymous"),
            "P001",
        )
        self.assertEqual(
            n170_task.sanitize_bids_label("run-01", "run", "01"),
            "01",
        )

    def test_design_matches_erp_core_defaults(self):
        report = n170_task.validate_design(seed=170)
        self.assertEqual(n170_task.READY_TEXT_HEIGHT_DEG, 0.72)
        self.assertEqual(report["practice_trials"], 10)
        self.assertEqual(report["main_trials"], 320)
        self.assertEqual(report["blocks"], 8)
        self.assertEqual(report["trials_per_block"], 40)
        self.assertEqual(report["trials_per_condition_per_block"], 10)
        self.assertEqual(set(report["condition_trial_counts"].values()), {80})
        self.assertEqual(report["stimulus_frames"], 18)
        self.assertEqual(report["isi_frame_range"], [66, 78])
        self.assertEqual(report["minimum_break_duration_s"], 20)
        self.assertTrue(report["break_can_extend_until_ready"])

    def test_every_main_block_is_balanced(self):
        trials = n170_task.build_main_trials(random.Random(170), "up", "down")
        for block_number in range(1, n170_task.BLOCKS + 1):
            block = [
                trial for trial in trials if trial.block_number == block_number
            ]
            self.assertEqual(len(block), n170_task.TRIALS_BETWEEN_BREAKS)
            for condition in n170_task.CONDITIONS:
                self.assertEqual(
                    sum(
                        trial.condition_name == condition.condition_name
                        for trial in block
                    ),
                    n170_task.TRIALS_PER_CONDITION_PER_BLOCK,
                )

    def test_main_images_each_repeat_twice(self):
        trials = n170_task.build_main_trials(random.Random(170), "up", "down")
        for condition in n170_task.CONDITIONS:
            condition_trials = [
                trial for trial in trials if trial.condition_name == condition.condition_name
            ]
            counts = {
                path: sum(trial.stimulus_path == path for trial in condition_trials)
                for path in n170_task.condition_files(condition)
            }
            self.assertEqual(set(counts.values()), {2})

    def test_each_image_appears_once_before_the_second_cycle(self):
        trials = n170_task.build_main_trials(random.Random(170), "up", "down")
        blocks_per_cycle = n170_task.BLOCKS // 2
        for cycle_number in (0, 1):
            first_block = cycle_number * blocks_per_cycle + 1
            last_block = first_block + blocks_per_cycle
            cycle = [
                trial
                for trial in trials
                if first_block <= trial.block_number < last_block
            ]
            for condition in n170_task.CONDITIONS:
                condition_trials = [
                    trial
                    for trial in cycle
                    if trial.condition_name == condition.condition_name
                ]
                self.assertEqual(
                    {trial.stimulus_path for trial in condition_trials},
                    set(n170_task.condition_files(condition)),
                )

    def test_trigger_code_ranges(self):
        trials = n170_task.build_main_trials(random.Random(170), "up", "down")
        expected = {
            "Faces": (1, 40),
            "Cars": (41, 80),
            "Scrambled Faces": (101, 140),
            "Scrambled Cars": (141, 180),
        }
        for name, expected_range in expected.items():
            codes = [trial.port_code for trial in trials if trial.condition_name == name]
            self.assertEqual((min(codes), max(codes)), expected_range)


if __name__ == "__main__":
    unittest.main()
