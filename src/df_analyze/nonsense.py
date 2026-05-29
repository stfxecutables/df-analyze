import logging
import warnings


_LIGHTNING_LOGGERS = (
    "lightning.pytorch.accelerators.cuda",
    "lightning.pytorch.utilities.rank_zero",
    "pytorch_lightning.accelerators.cuda",
    "pytorch_lightning.utilities.rank_zero",
)

_LIGHTNING_INFO_NOISE = (
    "LOCAL_RANK:",
    "CUDA_VISIBLE_DEVICES:",
    "available:",
)


class _LightningInfoFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(noise in msg for noise in _LIGHTNING_INFO_NOISE)


def silence_lightning_info() -> None:
    """Hide Lightning device banners while preserving warnings and errors."""
    warnings.filterwarnings(
        "ignore",
        message=r".*isinstance\(treespec, LeafSpec\).*deprecated.*",
        category=Warning,
        module=r"lightning\.pytorch\.utilities\._pytree",
    )
    for name in _LIGHTNING_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)
        if not any(isinstance(f, _LightningInfoFilter) for f in logger.filters):
            logger.addFilter(_LightningInfoFilter())


def silence_spam() -> None:
    logging.captureWarnings(capture=True)
    logger = logging.getLogger("py.warnings")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())
    logger.addFilter(lambda record: "UserWarning: Lazy modules" not in record.getMessage())


def enable_spam() -> None:
    logging.captureWarnings(capture=False)
