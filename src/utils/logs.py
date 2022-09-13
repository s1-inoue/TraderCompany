import datetime
import logging
import os


def get_mainlogger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    dt = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    os.makedirs("./logs/", exist_ok=True)
    handler1 = logging.FileHandler(f"./logs/{dt}.log")
    handler1.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(levelname)s, %(asctime)s, %(module)s, %(funcName)s, %(message)s"
    )
    handler1.setFormatter(fmt)

    logger.addHandler(handler1)
    return logger


def dump_history():
    pass


main_logger = get_mainlogger()
