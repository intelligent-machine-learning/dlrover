# Experiment Plan
  - [x] CDF图 **DDL: 9月5日**
  XPUTimer要使用CUPTI版本的
    - [x] FSDP:
    - [x] Megatron:
    - [x] DLRM:
  - [ ] 使用FSDP、Megatron、DLRM构造任务100个 **DDL: 9月8日前构造完脚本，9月12日前跑完50个，9月16日前跑完100个**
    20个有问题的，80个没有问题的，80个没问题的可以直接跑，然后看CDF、空泡率有没有不正常的，这种就是假阳性。
    - [ ] FSDP
      - [ ] 模型大小, 7B, 13B, 70B
      - [ ] GPU规模, 8, 16, 32, 48, 64
    - [ ] Megatron
      - [ ] 模型大小, 7B, 13B, 70B
      - [ ] GPU规模, 8, 16, 32, 48, 64
    - [ ] DLRM
      - [ ] 模型大小,
      - [ ] GPU规模,
      10

|       GPU        |  8   |  16  |  32  |  64  |
| :--------------: | :--: | :--: | :--: | :--: |
|     Megatron     |  8   |  8   |  8   |  4   |
|       FSDP       |  8   |  8   |  8   |  4   |
|       DLRM       |  8   |  8   |  8   |  0   |
| Megatron-badsync |  1   |  1   |  1   |  1   |
|     FSDP-Mem     |  1   |  1   |  1   |  1   |
|   FSDP-OPshape   |  1   |  1   |  1   |  1   |
| FSDP-Dataloader  |  1   |  1   |  1   |  1   |
|   DLRM-badsync   |  1   |  1   |  1   |  1   |

  - [ ] Greyhound: [https://github.com/wutianyuan1/Greyhound](https://github.com/wutianyuan1/Greyhound) **DDL: 9月12日前**
    - [ ] 首先是overhead，这里的XPU timer使用CUPTI版本
    - [ ] 其次是能不能检测慢
    - [ ] 以及FSDP、megatron、DLRM的适配性
