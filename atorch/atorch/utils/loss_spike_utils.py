""" loss spike utils
Save loss spike to files;
Decode loss spike and save Corresponding sample to a file. Using doc see ../docs/README-LOSS-SPIKE-UTIL.md
"""
import datetime
import os

import numpy as np

from atorch.common.log_utils import default_logger as logger


class LossSpikeBase:
    def __init__(
        self,
        loss_spike_save_dir,
        sample_data_paths,
        each_sample_len,
        min_iter,
        min_loss,
        loss_info_splitter="\t",
        loss_sample_str_splitter=",",
    ):
        """
        init params
        Args:
            loss_spike_save_dir: str, The directory where loss spike files are stored
            sample_data_paths: The path information stored in the sample can be a user-defined structure,
                such as name + path, tuple list:
                [("wikipedia", "corpus/base"), ("zhihu", "/dataset/fd5061f6/data/tokenize_data/zhihu.lazy")]
            each_sample_len: int, The length of a single sample
            min_iter: int, Record the minimum iterations required for loss
            min_loss: float, Record the minimum loss threshold required for loss
            loss_info_splitter: str, Delimiter used to store loss spike information, default ='\t'
            loss_sample_str_splitter: str, Delimiter used by str for a batch of loss or sample information, default =','
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
    def save_loss(self, file_name, cur_loss, cur_iter, *args, **kargs):
        """
        Store spike loss and corresponding information
        Args:
            file_name: str, loss spike file name
            cur_loss: float, current avg loss value
            cur_iter: int, current iteration
            args/kargs: any custom data in string format.

        """
        file_path = os.path.join(self.loss_spike_save_dir, file_name)
        losses_str = kargs["losses_str"]
        sample_infos_str = kargs["sample_infos_str"]
        if cur_loss > self.min_loss and cur_iter > self.min_iter:
            logger.info(f"save loss={cur_loss}, iter={cur_iter}")
            # define structure
            cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            info = self.loss_info_splitter.join([cur_time, str(cur_iter), str(cur_loss), losses_str, sample_infos_str])
            info = info + "\n"
            with open(file_path, "a+") as w:
                w.write(info)

    def decode_loss_spike(self, result_file_path, tokenizer, min_iter=None, min_loss=None):
        """
        According to the information such as spike loss in the file, the corresponding sample is parsed to the file
        Args:
            result_file_path: str, The address of the file that stores the contents of the decoded sample
            tokenizer: instance,
            min_iter: int, minimum iterations required for decode loss、sample
            min_loss: float,  minimum loss required for decode loss、sample

        Returns:

        """
        if min_iter is None:
            min_iter = self.min_iter
        if min_loss is None:
            min_loss = self.min_loss

        with open(result_file_path, "w") as fw:
            # Traverse the entire directory
            for loss_spike_file in os.listdir(self.loss_spike_save_dir):
                file_path = os.path.join(self.loss_spike_save_dir, loss_spike_file)
                with open(file_path) as fr:
                    for line in fr:
                        # process file content by line
                        # structure: f"{ctime}\t{iteration}\t{loss}\t{loss_str}\t{sample_ids_str}\n"
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
        losses = [float(e) for e in losses_str.split(self.loss_sample_str_splitter)]
        sample_infos = sample_infos_str.split(self.loss_sample_str_splitter)
        if len(losses) != len(sample_infos):
            logger.warn("batch loss length != batch sample length")
            return None, None, None

        losses = np.array(losses)
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
        # Here is more customized, different application scenarios can build different subclasses
        # 20-17-1385697-14158189-936633
        # {scatter_id}-{dsid}-{idx}-{raw_id}-{sample_id}
        scatter_id, dsid, _, _, sample_id = each_sample_info.split("-")  # structure is defined during initialization

        # Using the index of the list, 0 is the train data and 1 is the corresponding sample PATH
        ds_info = self.sample_data_paths[int(dsid)]

        datapath = f"{ds_info[1]}.scatter/{scatter_id}.lazy/text"

        if not os.path.exists(datapath):
            logger.warn("sample data path not exist")
            return None, None
        flen = self.get_data_file_len(datapath, np.dtype(np.int32))
        sample_cnt = flen // self.each_sample_len
        f = np.memmap(datapath, dtype=np.int32, shape=(sample_cnt, self.each_sample_len))  # Disk to memory
        data = f[int(sample_id)]
        return ds_info[0], data
