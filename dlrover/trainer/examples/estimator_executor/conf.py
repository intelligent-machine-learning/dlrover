from dlrover.trainer.tensorflow.util.column_info import Column
from MyEstimator import MyEstimator

def compare_fn(prev_eval_result, cur_eval_result):
    return True, {"fake_metric":0.9}

# 编写配置类, 类名请勿修改
class TrainConf(object):
    classifier_class = MyEstimator # estimator 模型
    batch_size = 64 # training 和 evaluation 所用的 batch size
    log_steps = 100 # 每训练多少个 steps 打印一条训练信息
    save_steps = 1000 # 每训练多少个 global step 触发一次 validation
    save_min_secs = 60  # 至少要等待多少秒才触发一次 validation，跟 save_step 是 and 关系
    save_max_secs = 60*6 # 过了多少秒必触发一次 validation，跟上面两个是 or 关系

	# 传到 estimator 模型中的的参数
    params = {
        "deep_embedding_dim": 8,
        "learning_rate": 0.0001,
        "l1": 0.0,
        "l21": 0.0,
        "l2": 0.0,
        "optimizer": "group_adam",
        "log_steps": 100
    }
	# 配置 odps 训练数据
    train_set = {
        "path": "file://test.data",
        "columns": (
            Column.create(  # type: ignore
                name="x",
                dtype="float32",
                is_label=False,
            ),
            Column.create(  # type: ignore
                name="y",
                dtype="float32",
                is_label=True,
            ),
        ),
    }
