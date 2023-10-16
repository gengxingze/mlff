from enum import Enum
import logging


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3
    ROOT = 4


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.root = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.root = self.avg ** 0.5


def setup_logger():
    # 创建日志记录器
    logger = logging.getLogger('MLFF')
    logger.setLevel(logging.DEBUG)

    # 创建文件处理器
    file_handler = logging.FileHandler('log', encoding="utf-8")

    # 设置日志输出格式
    formatter = logging.Formatter('[%(levelname)s]- %(message)s', datefmt='%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)

    # 添加文件处理器到记录器
    logger.addHandler(file_handler)

    return logger
