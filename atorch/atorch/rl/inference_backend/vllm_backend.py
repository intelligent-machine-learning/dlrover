import torch

from atorch.common.log_utils import default_logger as logger

try:
    from vllm import LLM, SamplingParams
except Exception:
    logger.warning("vllm not installed")


class VLLMComm:
    def __init__(self, ip, port):
        from vllm_comm import vllm_comm

        self.client = vllm_comm.vllmClient(ip, port)
        self.client.create_session()

    def send_data(self, data):
        self.client.send_data(data.data_ptr(), data.numel() * 2)
        self.client.delete_session()
        self.client.create_session()


class VLLMBackend:
    def __init__(self, checkpoint_path=None, gen_kwargs=None, gpu_memory_utilization=0.4, dtype="bfloat16"):
        self.gen_kwargs = gen_kwargs
        max_tokens = self.gen_kwargs.get("max_new_tokens", 500)
        self.sampling_params = SamplingParams(temperature=0, max_tokens=max_tokens)
        self.llm = LLM(
            model=checkpoint_path,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            dtype=dtype,
        )
        for param in self.llm.llm_engine.workers[0].model.parameters():
            param.data = torch.empty(0, dtype=param.dtype, device=param.device)
        torch.cuda.empty_cache()
        self.tokenizer = None

    def set_train_model_weights(self, train_model):
        for p1, p2 in zip(self.llm.llm_engine.workers[0].model.parameters(), train_model.parameters()):
            p1.data = p2.data

    def set_tokenizer(self, tokenizer):
        self.llm.llm_engine.tokenizer = tokenizer

    def generate(self, prompts):
        return self.llm.generate(prompts, sampling_params=self.sampling_params)
