import logging


class PickleableLogger(logging.Logger):
    def __init__(self, logger_name, fout_name, verbose):
        super().__init__(logger_name)
        self.setLevel(logging.DEBUG)

        self.logger_name = logger_name
        self.fout_name = fout_name
        self.verbose = verbose

        self._init()

    def _init(self):
        # File handler always in debug mode
        fh = logging.FileHandler(self.fout_name)
        ##fh.setLevel(logging.DEBUG)

        # STDOUT
        ch = logging.StreamHandler()

        if self.verbose:
            ch.setLevel(logging.DEBUG)
            fh.setLevel(logging.DEBUG)
        else:
            ch.setLevel(logging.INFO)
            fh.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        self.addHandler(fh)
        self.addHandler(ch)

    def __reduce__(self):
        return (PickleableLogger, (self.logger_name, self.fout_name, self.verbose))


def create_logger(logger_name, fout_name, verbose=False):
    ret = PickleableLogger(logger_name, fout_name, verbose)
    return ret
