import unittest

from transformers import LlamaForCausalLM

from atorch.rl.inference_backend.vllm_backend import VLLMBackend


@unittest.skipIf(True, "need weights and tokenizer")
class TestVllmInferenceBackEnd(unittest.TestCase):
    def test_vllm_backend_generate(self):
        checkpoint_path = "/mnt1/xuantai.hxd/test/llama2_small"
        # to do add param type
        vllm_inference_backend = VLLMBackend(checkpoint_path=checkpoint_path, gen_kwargs={})
        model = LlamaForCausalLM.from_pretrained(checkpoint_path)
        model.half().to(0)
        vllm_inference_backend.set_train_model_weights(model)


if __name__ == "__main__":
    unittest.main()
