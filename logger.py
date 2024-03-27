import logging
import sys
from collections import ChainMap
from typing import Any, Dict, List, Tuple

import wandb

from args import TrainerArguments


__all__ = (
    'MetricTracker',
    'BestMetricTracker',
    'create_logger',
    'log',
    'log_metric'
)


class MetricTracker:
    name: str
    prefix: str
    value: Any

    def __init__(self, name: str, prefix: str = None) -> None:
        self.prefix = prefix
        self.name = name

    def update(self, val: Any) -> None:
        self.value = val

    def format_dict(self) -> Dict[str, Any]:
        if self.prefix:
            return {f"{self.prefix}/{self.name}": self.value}
        else:
            return {f"{self.name}": self.value}


class BestMetricTracker(MetricTracker):
    metric: "BestMetric"

    class BestMetric:
        def __init__(self, value, epoch) -> None:
            self.value = value
            self.epoch = epoch

        def __eq__(self, other: "BestMetricTracker.BestMetric") -> bool:
            return self.value == other.value

        def __lt__(self, other: "BestMetricTracker.BestMetric") -> bool:
            return self.value < other.value

        def get(self, round_digits: int = 4) -> Tuple[int, int]:
            return int(self.value * (10 ** round_digits)), self.epoch

    def __init__(
        self,
        name: str,
        prefix: str = None,
    ) -> None:
        super().__init__(name, prefix)
        self.epoch = 0

    def update(self, val: float, epoch: int) -> None:
        this_val = self.metric.value if "metric" in self.__dict__ else 0.
        if this_val < val:
            self.metric = BestMetricTracker.BestMetric(val, epoch)

    @property
    def value(self) -> float:
        return self.metric.value


class Logger:
    def log(self, msg: str) -> None:
        raise NotImplementedError

    def log_metric(self, *metrics: Tuple[MetricTracker], **kwargs) -> None:
        metric_dict = dict(ChainMap(*[metric.format_dict() for metric in metrics]))
        self._log_metric(metric_dict, **kwargs)

    def _log_metric(self, metric_dict: Dict[str, Any], **kwargs) -> None:
        raise NotImplementedError


class WandbLogger(Logger):
    def __init__(self) -> None:
        pass

    def log(self, msg: str) -> None:
        pass

    def _log_metric(self, metric_dict: Dict[str, Any], **kwargs) -> None:
        wandb.log(metric_dict, **kwargs)


class SysLogger(Logger):
    def __init__(self, log_path: str) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s |  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.FileHandler(filename=log_path, mode='w'),
                logging.StreamHandler(stream=sys.stdout),
            ],
        )

        self.logger = logging.getLogger(__name__)

    def log(self, msg: str) -> None:
        self.logger.info(msg)

    def _log_metric(self, metric_dict: Dict[str, Any], **kwargs) -> None:
        self.logger.info(metric_dict)


loggers: List[Logger] = []


def create_logger(args: TrainerArguments):
    global loggers
    loggers.append(SysLogger(args.log_file_path))
    if args.use_wandb:
        wandb.init(
            dir=args.log_dir,
            config=args,
            project=args.project,
            entity=args.entity,
            name=args.experiment_name
        )
        loggers.append(WandbLogger())


def log(msg: str):
    for logger in loggers:
        logger.log(msg)


def log_metric(*metrics: Tuple[MetricTracker], **kwargs):
    for logger in loggers:
        logger.log_metric(*metrics, **kwargs)
