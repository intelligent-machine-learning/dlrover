import os
import unittest

from dlrover.client.platform.ray.ray_job_submitter import (
    create_scheduler,
    load_conf,
)


class RayJobSubmitter(unittest.TestCase):
 

    def test_parse_conf(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        conf_path = os.path.join(current_directory, "demo.yaml")
        all_data = load_conf(conf_path)
        self.assertEqual(all_data["dashboardUrl"], "localhost:5000")
        submitter = create_scheduler(conf_path)
        job_id = submitter.submit()
        self.assertNotEqual(job_id, None)
        submitter.wait_until_finish(job_id)


if __name__ == "__main__":
    unittest.main(verbosity=2)