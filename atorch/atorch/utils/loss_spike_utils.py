""" loss spike utils
Save loss spike to files;
Decode loss spike and save Corresponding sample to a file.
"""
import os
import datetime

import numpy as np
from atorch.common.log_utils import default_logger as logger


class LossSpikeBase:
    def __init__(self, loss_spike_save_dir, sample_data_paths, each_sample_len, min_iter, min_loss,
                 loss_info_splitter='\t', loss_sample_str_splitter=','):
        """
        init params
        Args:
            loss_spike_save_dir: str, The directory where loss spike files are stored
            sample_data_paths: 样本存储的路径信息, 可以是用户自定义的结构，如名称+路径的tuple list:
                [("wikipedia", "corpus/base"), ("zhihu", "/dataset/fd5061f6/data/tokenize_data/zhihu.lazy")]
            each_sample_len: int, 单个样本的长度
            min_iter: int, 记录loss所需的最小迭代轮数
            min_loss: float, 记录loss所需的最小loss阈值
            loss_info_splitter: str, 存储loss尖刺信息时所用的分隔符, 默认值='\t'
            loss_sample_str_splitter: str, 一批loss或者sample信息组成的str所用的分隔符, 默认值=','
        """
        self.loss_spike_save_dir = loss_spike_save_dir
        self.sample_data_paths = sample_data_paths
        self.each_sample_len = each_sample_len
        self.min_iter = min_iter
        self.min_loss = min_loss
        self.loss_info_splitter = loss_info_splitter
        self.loss_sample_str_splitter = loss_sample_str_splitter

        if not os.path.exists(loss_spike_save_dir):
            raise ValueError("Param loss_spike_save_dir not exist!")
        logger.info("Loss spike init success")

    @staticmethod
    def get_data_file_len(fpath, dtype):
        with open(fpath) as f:
            f.seek(0, 2)
            return f.tell() // dtype.itemsize


class TokenLossSpike(LossSpikeBase):
    def save_loss(self, file_name, cur_loss, cur_iter, losses_str, sample_infos_str):
        """
        存储尖刺loss及对应的一些信息
        Args:
            file_name: str, loss spike file name
            cur_loss: float, current loss value
            cur_iter: int, current iteration
            losses_str: A string of loss concatenated with splitter, 如:
                "2.31,2.30,10.98,1.56"
            sample_infos_str: A string of sample id info concatenated with splitter, 如:
                "20-17-1385697-14158189-936633,20-17-1385697-14158189-936633"

        """
        file_path = os.path.join(self.loss_spike_save_dir, file_name)
        if cur_loss > self.min_loss and cur_iter > self.min_iter:
            logger.info(f"save loss={cur_loss}, iter={cur_iter}")
            # 存储结构需要定义
            cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            info = self.loss_info_splitter.join([cur_time, str(cur_iter), str(cur_loss), losses_str, sample_infos_str])
            info = info + "\n"
            with open(file_path, 'a+') as w:
                w.write(info)

    def decode_loss_spike(self, result_file_path, tokenizer, min_iter=None, min_loss=None):
        """
        根据文件中的尖刺loss等信息，解析出对应的样本到文件
        Args:
            result_file_path: str, 存储解码后的样本内容的文件地址
            tokenizer: instance, 样本文件中的存储的样本, 需要时的解码器
            min_iter: int, 解码loss、sample所需的最小迭代轮数, 不传的话，使用存储时的初始化的阈值
            min_loss: float, 解码loss、sample所需的最小loss阈值, 不传的话，使用存储时的初始化的阈值

        Returns:

        """
        if min_iter is None:
            min_iter = self.min_iter
        if min_loss is None:
            min_loss = self.min_loss

        with open(result_file_path, "w") as fw:
            #  遍历整个目录
            for loss_spike_file in os.listdir(self.loss_spike_save_dir):
                file_path = os.path.join(self.loss_spike_save_dir, loss_spike_file)
                with open(file_path) as fr:
                    for line in fr:
                        # process file content by line
                        # 内容结构: f"{ctime}\t{iteration}\t{loss}\t{loss_str}\t{sample_ids_str}\n"
                        fcontent = line.strip().split(self.loss_info_splitter)
                        cur_iter = int(fcontent[1])
                        cur_loss = float(fcontent[2])
                        loss_str = fcontent[3]
                        sample_infos_str = fcontent[4]
                        if cur_iter < min_iter or cur_loss < min_loss:
                            logger.info(f"The content with iter={cur_iter} and loss={cur_loss} will not be parsed!")
                            continue
                        # parse
                        logger.info(f"Parse content with iter={cur_iter} and loss={cur_loss}!")
                        ds, text, max_loss = self.parse_sample_content(loss_str, sample_infos_str, tokenizer)
                        if ds is None:
                            continue
                        fw.write(f"=========={ds}  {max_loss}================\n")
                        fw.write(f"{text}\n\n\n\n")

    def parse_sample_content(self, losses_str, sample_infos_str, tokenizer):
        losses = [float(e) for e in losses_str.split(self.loss_sample_str_splitter)]  # 解析loss es
        sample_infos = sample_infos_str.split(self.loss_sample_str_splitter)
        if len(losses) != len(sample_infos):
            logger.warn("batch loss length != batch sample length")
            return None, None, None

        losses = np.array(losses)  # 解析loss es
        idx = losses.argmax(-1)
        max_loss = losses[idx]
        sample_with_max_loss = sample_infos[idx]
        ds, data = self.fetch(sample_with_max_loss)
        if ds is None:
            return None, None, None
        if tokenizer is not None:
            data = tokenizer.decode(data)
        return ds, data, max_loss

    def fetch(self, each_sample_info):
        # 这里是比较定制化的，不同应用场景可以构建不同的子类
        # 20-17-1385697-14158189-936633
        # {scatter_id}-{dsid}-{idx}-{raw_id}-{sample_id}
        scatter_id, dsid, _, _, sample_id = each_sample_info.split('-')  # 这个结构是在初始化定义好的

        ds_info = self.sample_data_paths[int(dsid)]  # 用列表的索引？0是train data，1是对应的NAMED_CORPORA[e].PATH

        datapath = f'{ds_info[1]}.scatter/{scatter_id}.lazy/text'  # 这怎么办啊，太定制了

        if not os.path.exists(datapath):
            logger.warn("sample data path not exist")
            return None, None
        flen = self.get_data_file_len(datapath, np.dtype(np.int32))
        sample_cnt = flen // self.each_sample_len
        f = np.memmap(datapath, dtype=np.int32, shape=(sample_cnt, self.each_sample_len))  # 磁盘写内存
        data = f[int(sample_id)]
        return ds_info[0], data
