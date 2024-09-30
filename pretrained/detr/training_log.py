from util.plot_utils import plot_logs
from pathlib import Path


if __name__ == '__main__':

    log_directory = [Path('outputs/')]

    fields_of_interest = ('loss', 'mAP')
    plot_logs(log_directory, fields_of_interest)

    fields_of_interest = ('loss_ce', 'loss_bbox', 'loss_giou')
    plot_logs(log_directory, fields_of_interest)

    fields_of_interest = ('class_error', 'cardinality_error_unscaled')
    plot_logs(log_directory, fields_of_interest)
