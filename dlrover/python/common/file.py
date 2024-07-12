from typing import List
from dlrover.python.common.log import default_logger as logger


def read_last_n_lines(filepath: str, n_lines: int) -> List[str]:
    try:
        with open(filepath, 'rb') as f:
            remain_lines = n_lines
            block_size = 1024
            buffer = bytearray()

            f.seek(0, 2)
            total_size = f.tell()

            while remain_lines > 0 and total_size > 0:
                read_size = min(block_size, total_size)
                f.seek(-read_size, 1)
                buffer.extend(f.read(read_size))
                total_size -= read_size
                f.seek(-read_size, 1)

                remain_lines -= buffer.count(b'\n')

            lines = buffer.split(b'\n')
            return lines[-n_lines:]
    except Exception as e:
        logger.error(f"fail to read {n_lines} line from {filepath}: {e}")
        return []
