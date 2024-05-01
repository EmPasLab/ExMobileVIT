# Import basic pkg
import os
from pytorch_lightning import loggers as pl_loggers

# Set Logger
def get_logger(args):

    # Logger for pytorch lightning

    # CSV Logger for pytorch lightning
    CSVLogger = pl_loggers.CSVLogger(save_dir = args.log_dir,
                                    name = args.log_name,
                                    version = args.version,
                                    flush_logs_every_n_steps = args.log_freq,)

    # Tensorboard Logger for pytorch lightning
    TB_logger = pl_loggers.TensorBoardLogger(save_dir = args.log_dir,
                                    name = args.log_name,
                                    version = args.version,
                                    default_hp_metric=False)

    return CSVLogger, TB_logger