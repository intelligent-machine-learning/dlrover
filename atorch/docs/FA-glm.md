# Flash Attention GLM-Mask Support

ATorch provides Flash Attention, FA for short, with features of [GLM](https://arxiv.org/abs/2103.10360) style mask. 

Currently we support [fa1_additive_mask](https://github.com/intelligent-machine-learning/flash-attention/tree/fa1_additive_mask), [fa2_glm_mask](https://github.com/intelligent-machine-learning/flash-attention/tree/fa2_glm_mask) and [fa2_pack_glm_mask](https://github.com/intelligent-machine-learning/flash-attention/tree/fa2_pack_glm_mask).

## Installation

- fa1_additive_mask
    - Renamed to new lib `flash_attn_1` for maintaining two version of FA in one image environment.
    - Referring to [PR57](https://github.com/Dao-AILab/flash-attention/pull/57)

```shell
git clone -b fa1_additive_mask https://github.com/intelligent-machine-learning/flash-attention
cd flash-attention
python setup.py bdist_wheel  # take around 10 min
pip install dist/flash_attn_1-0.2.6.post2-*.whl
```

- fa2_glm_mask

```shell
git clone -b fa2_glm_mask https://github.com/intelligent-machine-learning/flash-attention
cd flash-attention
FLASH_ATTN_LOCAL_VERSION=glm-mask MAX_JOBS=80 python setup.py bdist_wheel 
pip install dist/flash_attn-2.0.4+glm.mask-*.whl
```

- or fa2_pack_glm_mask

```shell
git clone -b fa2_pack_glm_mask https://github.com/intelligent-machine-learning/flash-attention
cd flash-attention
FLASH_ATTN_LOCAL_VERSION=pack-glm-mask MAX_JOBS=80 FLASH_ATTENTION_FORCE_BUILD=TRUE python setup.py bdist_wheel 
pip install dist/flash_attn-2.3.6+pack.glm.mask-*.whl
```

## Usage

- For fa1_additive_mask, `flash_attn_xxx_func`s in `flash_attn_1.flash_attn_interface` add `attn_mask` and `attn_bias` kwargs.

```
We use the following notation:
        batch_size: n
        sequence_length: s_q, s_k
        nh: number of attention heads
        hs: head dimension
attn_mask: [b, nh or 1, s_q or 1, s_k]
attn_bias: [1, nh, s_q, s_k]
```

- For `fa2_glm_mask` or `fa2_pack_glm_mask`, `flash_attn_xxx_func`s in `flash_attn.flash_attn_interface` add `glm_mask` kwarg.
    - Refer to [glm mask format](#glm-mask-format) for more information.

```
# fa2_glm_mask
glm_mask: [batch_size] in torch.int32.
# fa2_pack_glm_mask
glm_mask: [batch_size, 2, MAX_NUM_PAIR] in torch.int32. Compat [batch_size].
```

- ATorch provides [`FlashAttnModule`](https://github.com/intelligent-machine-learning/dlrover/blob/458484308b32aed8590cc5d5c0011c25dae0f6f2/atorch/atorch/modules/transformer/layers.py#L1373) as unified API for FlashAttention, make it a module for train/val dropout_p handling.
    - support key_padding_mask; support additive mask/bias in FA1; support break_point index glm-mask in FA2; support startpoint/endpoint pack-glm-mask in FA2.

```python
from atorch.modules.transformer.layers import FlashAttnModule

# in __init__
self.FA = FlashAttnModule(causal=causal, softmax_scale=softmax_scale, attention_dropout=attention_dropout)

# in forward
## fa1_additive_mask, init causal should be False
out = self.FA(q, k, v, additive_mask=additive_mask)
## fa2(_pack)_glm_mask, init causal should be True
out = self.FA(q, k, v, glm_mask=glm_mask)
```

<a id="glm mask format"></a>
## glm mask format

- [batch_size] glm_mask: semantically same to the index for building mask matrix in [modeling_glm.py](https://huggingface.co/THUDM/glm-10b/blob/696788d4f82ac96b90823555f547d1e754839ff4/modeling_glm.py#L538).

    - e.g. batch_size = 2, sequence length = 5.

```python
>>> glm_mask
tensor([2, 3], device='cuda:0', dtype=torch.int32)
>>> additive_glm_mask
tensor([[[[    -0.,     -0., -60000., -60000., -60000.],
          [    -0.,     -0., -60000., -60000., -60000.],
          [    -0.,     -0.,     -0., -60000., -60000.],
          [    -0.,     -0.,     -0.,     -0., -60000.],
          [    -0.,     -0.,     -0.,     -0.,     -0.]]],


        [[[    -0.,     -0.,     -0., -60000., -60000.],
          [    -0.,     -0.,     -0., -60000., -60000.],
          [    -0.,     -0.,     -0., -60000., -60000.],
          [    -0.,     -0.,     -0.,     -0., -60000.],
          [    -0.,     -0.,     -0.,     -0.,     -0.]]]], device='cuda:0',
       dtype=torch.float16)
>>> 
```

- [batch_size, 2, MAX_NUM_PAIR] packed glm_mask: startpoint/endpoint mean [startpoint, endpoint) intervals are bidirectional. -1 means invalid. MAX_NUM_PAIR doesn't have to be a constant, just make sure the shorter samples fill -1 to the longest sample.

    -   e.g. batch_size = 2, sequence length = 20.

```python
>>> pack_glm_mask
tensor([[[ 0,  5,  9],
         [ 3,  7, 19]],

        [[ 0,  9, -1],
         [ 6, 12, -1]]], device='cuda:0', dtype=torch.int32)
>>> (additive_pack_glm_mask == 0).int()
tensor([[[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # start
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # end
          [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # start
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # end
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0], # start
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]], # end
        #  s        e     s     e     s                             e


        [[[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # start
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # end
          [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # start
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0], # end
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]]],
       #   s                 e        s        e
       device='cuda:0', dtype=torch.int32)
>>> additive_pack_glm_mask.unique()
tensor([-60000.,     -0.], device='cuda:0')
```
