import time
import unittest

from atorch.common.log_utils import Timer, TimeStats


class LogUitlTests(unittest.TestCase):
    def test_timer(self):
        timer = Timer("test")
        timer.start()
        time.sleep(1)
        timer.end()
        self.assertAlmostEqual(timer.elapsed_time, 1, places=1)

        timestate = TimeStats("test")
        timestate[timer.name] = timer.elapsed_time
        with Timer("forward", timestate):
            time.sleep(1)

        self.assertAlmostEqual(timestate["forward"], 1, places=1)


if __name__ == "__main__":
    unittest.main()
