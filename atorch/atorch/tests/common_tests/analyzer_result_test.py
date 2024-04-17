import unittest

from atorch.auto.engine.analyser_result import AnalyserResult


class AnalyzerResultTest(unittest.TestCase):
    def test_analyzer_result(self):
        result = {
            "analyse_basic": {
                "model_params_num": {"layer_0": 10, "layer_1": 20},
                "model_params_mb": {"layer_0": 0.1, "layer_1": 0.2},
            },
            "analyse_dynamic": {"fixed_data_size": True, "data_size": 1024},
        }

        analyzer_result = AnalyserResult()
        analyzer_result.update(result)
        self.assertEqual(
            analyzer_result.get("analyse_basic"),
            {
                "model_params_num": {"layer_0": 10, "layer_1": 20},
                "model_params_mb": {"layer_0": 0.1, "layer_1": 0.2},
            },
        )

        self.assertEqual(
            analyzer_result.get("model_params_num"),
            {"layer_0": 10, "layer_1": 20},
        )

        self.assertEqual(analyzer_result.get("data_size"), 1024)
        self.assertEqual(analyzer_result.get("analyse_transformer"), None)


if __name__ == "__main__":
    unittest.main()
