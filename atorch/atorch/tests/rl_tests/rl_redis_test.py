import subprocess
import unittest

from atorch.common.log_utils import default_logger as logger
from atorch.rl.model_utils.redis_util import RMQ, RedisDataLoader

try:
    from vllm import LLM, SamplingParams  # type: ignore
except Exception:
    logger.warning("vllm not installed")


class TestDsLlama2Container(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # start redis server
        cls.redis_process = subprocess.Popen(["redis-server", "--port 6379"])

    @classmethod
    def tearDownClass(cls):
        # stop redis server
        cls.redis_process.terminate()
        cls.redis_process.wait()

    def test_redis_client(self):

        client = RMQ("test")

        model_dir = "/mnt1/xuantai.hxd/test/llama2_small"

        sampling_params = SamplingParams(temperature=0, max_tokens=500)

        llm = LLM(model=model_dir, trust_remote_code=True, gpu_memory_utilization=0.5, tensor_parallel_size=1)

        res = llm.generate("大家对待男女明星差别那么大?", sampling_params=sampling_params)
        import json

        data = {"prompt": res[0].prompt, "output": res[0].outputs[0].text}

        dataloader = RedisDataLoader()

        for i in range(200000):
            client.publish(json.dumps(data), str(0))

        client.publish("STOP", "0")

        for i, j in enumerate(dataloader):
            if i == 4:
                break
            assert j["prompt"][0] == "大家对待男女明星差别那么大?"


if __name__ == "__main__":
    unittest.main()
