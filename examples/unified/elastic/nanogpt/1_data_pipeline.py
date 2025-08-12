import numpy as np
import ray
import ray.data


def build_vocab(data):
    chars = sorted(list(set(data)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos, len(chars)


def encode_string(s, stoi):
    return [stoi[c] for c in s]


def ray_prepare_and_load_dataset(
    input_file_path, block_size=128, batch_size=32
):
    # 1. 读取原始文本
    with open(input_file_path, "r") as f:
        data = f.read()
    stoi, itos, vocab_size = build_vocab(data)
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data = data[int(n * 0.9) :]

    # 2. 用 Ray Data 构建数据集（每个样本为一个字符）
    train_ds = ray.data.from_items([{"ch": c} for c in train_data])
    val_ds = ray.data.from_items([{"ch": c} for c in val_data])

    # 3. 编码为 token id
    train_ds = train_ds.map_batches(
        lambda batch: {
            "id": np.array([stoi[c] for c in batch["ch"]], dtype=np.uint16)
        },
        batch_format="pandas",
    )
    val_ds = val_ds.map_batches(
        lambda batch: {
            "id": np.array([stoi[c] for c in batch["ch"]], dtype=np.uint16)
        },
        batch_format="pandas",
    )

    # 4. 滑动窗口切 block
    def blockify(batch, block_size):
        arr = batch["id"]
        blocks = []
        for i in range(len(arr) - block_size):
            x = arr[i : i + block_size]
            y = arr[i + 1 : i + 1 + block_size]
            if len(x) == block_size and len(y) == block_size:
                blocks.append({"x": x, "y": y})
        return blocks

    # train_ds = train_ds.window(block_size + 1, step=1).flat_map(
    #     lambda window: blockify(
    #         {"id": np.concatenate([row["id"] for row in window])}, block_size
    #     )
    # )
    # val_ds = val_ds.window(block_size + 1, step=1).flat_map(
    #     lambda window: blockify(
    #         {"id": np.concatenate([row["id"] for row in window])}, block_size
    #     )
    # )

    # 5. 返回 Ray Data 的 batch 迭代器
    train_iter = train_ds.iter_batches(
        batch_size=batch_size, batch_format="numpy"
    )
    val_iter = val_ds.iter_batches(batch_size=batch_size, batch_format="numpy")
    return train_iter, val_iter, vocab_size


# 用法示例
# train_iter, val_iter, vocab_size = ray_prepare_and_load_dataset("your_data.txt", block_size=128, batch_size=32)
# for batch in train_iter:
#     x, y = batch["x"], batch["y"]
#     # x, y 已经是 numpy 数组，可直接转 torch.Tensor
