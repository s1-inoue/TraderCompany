from cgitb import handler
import logging
import datetime


def get_mainlogger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    today = datetime.date.today()

    handler1 = logging.FileHandler(f"./logs/{today}.log")
    handler1.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(levelname)s, %(asctime)s, %(module)s, %(funcName)s, %(message)s"
    )
    handler1.setFormatter(fmt)

    logger.addHandler(handler1)
    logger.info("Logging start")
    return logger


def dump_history():
    pass


main_logger = get_mainlogger()
