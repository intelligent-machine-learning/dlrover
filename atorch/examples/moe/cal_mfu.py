class ModelUtil:
    """
    All FLOPs calculations consider only matrix multiplication.
    """

    def __init__(
        self,
        hidden_size,
        num_hidden_layers,
        num_attention_heads,
        intermediate_size,
        vocab_size,
        seq_length,
        batch_size,
        num_key_value_heads=None,
        use_gradient_checkpointing=False,
        use_gate_proj=False,
        moe=False,
        num_experts=1,
        top_k=1,
        shared_experts=0,
    ) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            self.num_key_value_heads = num_attention_heads
        else:
            self.num_key_value_heads = num_key_value_heads

        assert self.hidden_size % self.num_attention_heads == 0
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.kv_hidden_size = self.head_dim * self.num_key_value_heads

        self.seq_length = seq_length
        self.batch_size = batch_size

        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gate_proj = use_gate_proj

        self.moe = moe
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_experts = shared_experts

    def get_FLOPs_attention(self, num_layers: int = None):
        """
        Calculate FLOPs of Attention forward.

        Return:
            Model FLOPs
        """
        if num_layers is None:
            num_layers = self.num_hidden_layers

        q_projection = 2 * self.batch_size * self.seq_length * self.hidden_size**2
        kv_projection = 2 * 2 * self.batch_size * self.seq_length * self.hidden_size * self.kv_hidden_size
        qk_product = (
            2
            * self.batch_size
            * self.seq_length**2
            * (self.num_attention_heads * self.head_dim)  # Equal to self.hidden_size
        )
        v_product = 2 * self.batch_size * self.seq_length**2 * (self.num_attention_heads * self.head_dim)
        final_linear = 2 * self.batch_size * self.seq_length * self.hidden_size**2

        layer_norm = 2 * self.batch_size * self.seq_length * self.hidden_size

        attn_fwd_flops = q_projection + kv_projection + qk_product + v_product + final_linear + layer_norm

        return attn_fwd_flops * num_layers

    def get_FLOPs_mlp(self, num_layers: int = None):
        """
        Calculate FLOPs of MLP forward.

        Return:
            Model FLOPs
        """
        if num_layers is None:
            num_layers = self.num_hidden_layers

        if self.use_gate_proj:
            mlp_fwd_flops = 3 * 2 * self.batch_size * self.seq_length * self.hidden_size * self.intermediate_size
        else:
            mlp_fwd_flops = 2 * 2 * self.batch_size * self.seq_length * self.hidden_size * self.intermediate_size

        if self.moe:
            dense_flops = mlp_fwd_flops * (self.top_k + self.shared_experts)
            gate_flops = 3 * self.seq_length * self.num_experts

            mlp_fwd_flops = dense_flops + gate_flops

        return mlp_fwd_flops * num_layers

    def get_FLOPs_perlayer(self):
        """
        Calculate FLOPs of per Transformer Layer forward.

        Return:
            Model FLOPs
        """
        attn_fwd_flops = self.get_FLOPs_attention(num_layers=1)
        mlp_fwd_flops = self.get_FLOPs_mlp(num_layers=1)
        return attn_fwd_flops + mlp_fwd_flops

    def get_FLOPs_logits(self):
        """
        Calculate FLOPs of the final logits Linear forward.

        Return:
            Model FLOPs
        """
        logits_fwd_flops = 2 * self.batch_size * self.seq_length * self.hidden_size * self.vocab_size
        return logits_fwd_flops

    def get_FLOPs_total(self):
        """
        Calculate all FLOPs of Transformer model, include forward and backward.
        Just consider matrix multiplication.

        Return:
            (Model FLOPs, Hardware FLOPs)
        """
        attn_fwd_flops = self.get_FLOPs_attention()
        mlp_fwd_flops = self.get_FLOPs_mlp()
        logits_fwd_flops = self.get_FLOPs_logits()

        # Total FLOPs of forward
        fwd_flops = attn_fwd_flops + mlp_fwd_flops + logits_fwd_flops

        model_FLOPs = fwd_flops * 3  # bwd_flops = 2 * fwd_flops

        # Just for full gradient checkpointing
        hardware_FLOPs = fwd_flops * (4 if self.use_gradient_checkpointing else 3)

        return model_FLOPs, hardware_FLOPs

    def get_FLOPs_ratio_of_part(self):
        """
        Return the FLOPs ratio of each part.
        """
        res = {}

        layer_attn_fwd_flops = self.get_FLOPs_attention(num_layers=1)
        layer_mlp_fwd_flops = self.get_FLOPs_mlp(num_layers=1)
        layer_fwd_flops = self.get_FLOPs_perlayer()

        logits_fwd_flops = self.get_FLOPs_logits()

        total_fwd_flops = layer_fwd_flops * self.num_hidden_layers + logits_fwd_flops

        res["attn_in_layer"] = layer_attn_fwd_flops / layer_fwd_flops
        res["mlp_in_layer"] = layer_mlp_fwd_flops / layer_fwd_flops
        res["attn_in_total"] = layer_attn_fwd_flops * self.num_hidden_layers / total_fwd_flops
        res["mlp_in_total"] = layer_mlp_fwd_flops * self.num_hidden_layers / total_fwd_flops
        res["logits_in_total"] = logits_fwd_flops / total_fwd_flops
        return res

    def get_mem_perlayer(self):
        """
        Calculate GPU Memory footprint of per Transformer Layer.
        """
        pass

    def get_mem_total(self):
        """
        Calculate GPU memory footprint of Transformer model.
        """
        pass

    def get_communication_perlayer_with_tp(self):
        pass

    def get_communication_total(self):
        pass


# example for Llama2-7B on A100: use gated Linear and group attention, each iteration costs 8.868s
#
# python model_util.py -gp -b 64 -s 4096 -hs 4096 -v 32000 -i 11008 -l 32 -a 32 -kv 32 -nd 8 -c 8.868 -t 312.0
# output:
# Model TFLOP/s/GPU: 170.29, MFU: 0.5458, Hardware TFLOP/s/GPU: 170.29, HFU: 0.5458
# Calculation ratio of each part:
#     attn_in_layer:      0.4267
#     mlp_in_layer:       0.5733
#     attn_in_total:      0.4194
#     mlp_in_total:       0.5635
#     logits_in_total:    0.0171


def cal_mfu_simple(
    B,
    time,
    hidden_size,
    expert_intermediate_size,
    activated_expert_num,
    head_num,
    kv_head_num,
    sequence_length,
    layer_num,
    top_tflops,
):
    hidden_size_per_head = hidden_size // head_num
    # pre-attn kqv linear
    pre_attn_flops = 2 * sequence_length * hidden_size * (hidden_size + 2 * hidden_size_per_head * kv_head_num)
    # attn:
    atnn_flops = 2 * sequence_length * sequence_length * hidden_size * 2
    # post-attn linear
    post_attn_flops = 2 * sequence_length * hidden_size * hidden_size

    mlp_flops = activated_expert_num * sequence_length * (6 * hidden_size * expert_intermediate_size)

    per_layer = pre_attn_flops + atnn_flops + post_attn_flops + mlp_flops
    all_layer = layer_num * per_layer
    res = 3 * all_layer * B / time / top_tflops / 1e12

    return res


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch_size", type=int, default=32)
    parser.add_argument("-s", "--sequence_length", type=int, default=4096)
    parser.add_argument("-hs", "--hidden_size", type=int, default=4096)
    parser.add_argument("-v", "--vocab_size", type=int, default=32000)
    parser.add_argument("-i", "--intermediate_size", type=int, default=11008)
    parser.add_argument("-l", "--num_hidden_layers", type=int, default=32)
    parser.add_argument("-a", "--num_attention_heads", type=int, default=32)
    parser.add_argument(
        "-kv",
        "--num_key_value_heads",
        type=int,
        default=None,
    )
    parser.add_argument("-gp", "--use_gate_proj", action="store_true")
    parser.add_argument("-gc", "--use_gradient_checkpointing", action="store_true")
    parser.add_argument("-moe", "--moe", action="store_true")
    parser.add_argument("-ne", "--num_experts", type=int, default=1)
    parser.add_argument("-se", "--shared_experts", type=int, default=0, help="num of shared experts")
    parser.add_argument("-tk", "--top_k", type=int, default=1)
    parser.add_argument("-nd", "--num_devices", type=int)
    parser.add_argument("-c", "--cost", type=float, help="seconds per iteration")
    # 312.0 is A100's max TFLOPS
    parser.add_argument("-t", "--top_tflops", type=float, default=312.0)

    args = parser.parse_args()

    model_util = ModelUtil(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        seq_length=args.sequence_length,
        batch_size=args.batch_size,
        num_key_value_heads=args.num_key_value_heads,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gate_proj=args.use_gate_proj,
        moe=args.moe,
        num_experts=args.num_experts,
        top_k=args.top_k,
        shared_experts=args.shared_experts,
    )

    model_flops, hardware_flops = model_util.get_FLOPs_total()

    model_TFLOPS = model_flops / 1e12 / args.num_devices / args.cost
    hardware_TFLOPS = hardware_flops / 1e12 / args.num_devices / args.cost

    MFU = model_TFLOPS / args.top_tflops
    HFU = hardware_TFLOPS / args.top_tflops
    print(
        f"Model TFLOP/s/GPU: {model_TFLOPS:.2f}, MFU: {MFU:.4f}, "
        f"Hardware TFLOP/s/GPU: {hardware_TFLOPS:.2f}, HFU: {HFU:.4f}"
    )

    res = model_util.get_FLOPs_ratio_of_part()
    print("Calculation ratio of each part: ")
    for k, v in res.items():
        print(f"    {k}:\t{v:.4f}")

    activated_expert_num = args.top_k + args.shared_experts
    simple_res = cal_mfu_simple(
        B=args.batch_size,
        time=args.cost,
        hidden_size=args.hidden_size,
        expert_intermediate_size=args.intermediate_size,
        activated_expert_num=activated_expert_num,
        head_num=args.num_attention_heads,
        kv_head_num=args.num_key_value_heads,
        sequence_length=args.sequence_length,
        layer_num=args.num_hidden_layers,
        top_tflops=args.top_tflops,
    )
    print("Simple MFU:", simple_res)
