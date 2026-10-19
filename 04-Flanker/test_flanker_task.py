import random
import unittest

import flanker_task


class FlankerDesignTests(unittest.TestCase):
    def test_bids_labels_remove_prefixes_and_invalid_characters(self):
        self.assertEqual(
            flanker_task.sanitize_bids_label("sub-P 001", "sub", "anonymous"),
            "P001",
        )
        self.assertEqual(
            flanker_task.sanitize_bids_label("run-01", "run", "01"),
            "01",
        )

    def test_design_matches_erp_core_defaults(self):
        report = flanker_task.validate_design(seed=170)
        self.assertEqual(flanker_task.READY_TEXT_HEIGHT_DEG, 0.72)
        self.assertEqual(report["blocks"], 10)
        self.assertEqual(report["trials_per_block"], 40)
        self.assertEqual(report["practice_trials"], 10)
        self.assertEqual(report["main_trials"], 400)
        self.assertEqual(report["target_frames"], 12)
        self.assertEqual(report["isi_frame_range"], [72, 84])
        self.assertEqual(report["minimum_break_duration_s"], 20)
        self.assertTrue(report["break_can_extend_until_ready"])
        self.assertFalse(report["fixation_during_stimulus"])
        self.assertTrue(report["fixation_during_isi"])
        self.assertEqual(set(report["condition_trial_counts"].values()), {100})

    def test_every_main_block_is_balanced(self):
        trials = flanker_task.build_main_trials(random.Random(170))
        for block_number in range(1, flanker_task.BLOCKS + 1):
            for condition in flanker_task.CONDITIONS:
                count = sum(
                    trial.block_number == block_number
                    and trial.condition == condition
                    for trial in trials
                )
                self.assertEqual(
                    count,
                    flanker_task.TRIALS_PER_CONDITION_PER_BLOCK,
                )

    def test_stimulus_and_response_trigger_codes(self):
        expected = {
            "Congruent Left": (11, 111, 211),
            "Congruent Right": (12, 212, 112),
            "Incongruent Left": (21, 121, 221),
            "Incongruent Right": (22, 222, 122),
        }
        for condition in flanker_task.CONDITIONS:
            stimulus, correct, incorrect = expected[condition.name]
            self.assertEqual(condition.trigger_code, stimulus)
            self.assertEqual(
                flanker_task.response_trigger_code(
                    condition.correct_key,
                    condition.trigger_code,
                ),
                correct,
            )
            wrong_key = "right" if condition.correct_key == "left" else "left"
            self.assertEqual(
                flanker_task.response_trigger_code(wrong_key, condition.trigger_code),
                incorrect,
            )

    def test_feedback_thresholds_match_erp_core(self):
        self.assertEqual(
            flanker_task.feedback_for_error_rate(0.05),
            "Try to respond a bit faster.",
        )
        self.assertEqual(
            flanker_task.feedback_for_error_rate(0.15),
            "Great job!",
        )
        self.assertEqual(
            flanker_task.feedback_for_error_rate(0.25),
            "Try to respond more accurately.",
        )


if __name__ == "__main__":
    unittest.main()
