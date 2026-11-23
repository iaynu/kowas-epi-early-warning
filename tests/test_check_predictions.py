import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_predictions.py"


class CheckPredictionsTests(unittest.TestCase):
    def test_empty_csv_reports_header_error_without_traceback(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")
            result = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)

        self.assertEqual(result.returncode, 1)
        self.assertIn("헤더가 규정과 다릅니다", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
