from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def create_dataset(config):
    pass


class BaseDataSet:
    def __init__(self, prompts, max_prompt_length, tokenizer, padding=False, truncation=True, add_special_tokens=False):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.max_prompt_length = max_prompt_length
        self.padding = padding
        self.trunction = truncation
        self.add_special_tokens = add_special_tokens
        self.gen_prompts = []
        self.encode_prompts()
        self.generate_prompts(self.model_inputs)

    def encode_prompts(self):
        self.model_inputs = self.tokenizer(
            self.prompts,
            truncation=self.trunction,
            padding=self.padding,
            max_length=self.max_prompt_length,
            add_special_tokens=self.add_special_tokens,
        )

    def generate_prompts(self):
        pass

    def __getitem__(self, ix: int):
        return self.gen_prompts[ix]

    def __len__(self) -> int:
        return len(self.gen_prompts)

    def collate_fn(self):
        return DataCollatorWithPadding(self.tokenizer) if self.tokenizer else torch.vstack

    def create_loader(self, batch_size: int, shuffle=False) -> DataLoader:
        collate_fn = self.collate_fn()
        dataloader = DataLoader(self, batch_size=batch_size, collate_fn=collate_fn, shuffle=shuffle)
        return dataloader


class PromptDataset(BaseDataSet):
    """
    Tokenizes prompts, unless they are already tokenized,
    and truncates them to `max_prompt_length` from the right
    """

    def __init__(
        self,
        prompts,
        max_prompt_length,
        tokenizer,
        padding=False,
        truncation=True,
        add_special_tokens=False,
    ):
        super().__init__(
            prompts,
            max_prompt_length,
            tokenizer,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    def generate_prompts(self, model_inputs):
        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        self.gen_prompts = [
            {"input_ids": tokens[:-1], "attention_mask": mask[:-1]}
            for tokens, mask in zip(prompts_tokens, attention_mask)
        ]


class GLMPromptDataset(BaseDataSet):
    """
    Tokenizes prompts, unless they are already tokenized, and truncates them to `max_prompt_length` from the right
    """

    def __init__(self, prompts, max_prompt_length, tokenizer, padding=False, truncation=True, add_special_tokens=False):
        super().__init__(
            prompts,
            max_prompt_length,
            tokenizer,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    def generate_prompts(self, model_inputs):
        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]
        self.gen_prompts = [
            {"input_ids": tokens[:-1], "attention_mask": mask[:-1], "idx": idx}
            for idx, (tokens, mask) in enumerate(zip(prompts_tokens, attention_mask))
        ]


@dataclass
class PPORLElement:

    query_tensor: torch.tensor
    response_tensor: torch.tensor
    logprobs: torch.tensor
    values: torch.tensor
    rewards: torch.tensor


@dataclass
class PPORLBatch:
    query_tensors: torch.tensor
    response_tensors: torch.tensor
    logprobs: torch.tensor
    values: torch.tensor
    rewards: torch.tensor


PPORLElement_KEY = [k for k in PPORLElement.__annotations__.keys()]


class RLTrainingDataset:
    def __init__(self, replay_buffer):
        self.pad_token_id = None
        self.sop_token_id = None
        data = replay_buffer.data
        query_tensor = data["query_tensor"]
        response_tensor = data["response_tensor"]
        logprobs = data["logprobs"]
        values = data["values"]
        rewards = data["rewards"]
        self.ppo_rl_elements = [
            PPORLElement(
                query_tensor=i,
                response_tensor=j,
                logprobs=k,
                values=l,
                rewards=m,
            )
            for i, j, k, l, m in zip(query_tensor, response_tensor, logprobs, values, rewards)
        ]

    def __getitem__(self, ix: int):
        return self.ppo_rl_elements[ix]

    def __len__(self) -> int:
        return len(self.ppo_rl_elements)

    def set_tokenizer(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.sop_token_id = tokenizer.sop_token_id

    def collate_fn(self):
        def collate_fn(elems):
            assert self.pad_token_id is not None
            assert self.sop_token_id is not None
            remove_sop_pad = pad_sequence(
                [elem.query_tensor[:-1] for elem in elems],
                padding_value=self.pad_token_id,
                batch_first=True,
            )
            sops = torch.full((remove_sop_pad.size(0), 1), self.sop_token_id).to(remove_sop_pad.device)
            query_padded = torch.cat((remove_sop_pad, sops), -1)

            return PPORLBatch(
                # right padding of already right-padded queries
                query_padded,
                # Right pad the rest, to have a single horizontal query/response split
                pad_sequence(
                    [elem.response_tensor for elem in elems],
                    padding_value=self.pad_token_id,
                    batch_first=True,
                ),
                pad_sequence(
                    [elem.logprobs for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
                pad_sequence([elem.values for elem in elems], padding_value=0.0, batch_first=True),
                pad_sequence(
                    [elem.rewards for elem in elems],
                    padding_value=0.0,
                    batch_first=True,
                ),
            )

        return collate_fn

    def create_dataloader(
        self,
        batch_size: int,
        shuffle=False,
    ) -> DataLoader:

        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn())
