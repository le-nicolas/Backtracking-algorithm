import unittest

from backtracking_lab.linkage import generate_motion_targets, synthesize_fourbar


class LinkageSynthesisTests(unittest.TestCase):
    def test_hybrid_search_recovers_motion(self) -> None:
        input_angles = [20.0, 35.0, 50.0, 65.0, 80.0]
        target_angles = generate_motion_targets(
            input_angles,
            ground=4.0,
            crank=2.0,
            coupler=5.0,
            rocker=4.0,
            branch=1,
        )

        result = synthesize_fourbar(
            input_angles,
            target_angles,
            ground_values=[3.0, 4.0, 5.0],
            crank_values=[1.0, 2.0, 3.0],
            coupler_values=[4.0, 5.0, 6.0],
            rocker_values=[3.0, 4.0, 5.0],
            top_k=3,
            min_transmission_deg=5.0,
            refine=True,
            refine_steps=40,
        )

        self.assertGreaterEqual(result.stats.solutions_found, 1)
        self.assertGreaterEqual(len(result.candidates), 1)
        self.assertLess(result.candidates[0].rmse_deg, 0.2)


if __name__ == "__main__":
    unittest.main()
