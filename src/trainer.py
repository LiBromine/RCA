import os
import numpy as np
from common.args import parse_args
from common.utils import *
from src.detection import anomaly_detection


def main_worker(args):
    # init
    print("Data Load Begin.")
    data_kpis = load_kpis_from_csv(args.data_folder)
    data_times = load_times_from_csv(args.injection_times)
    print("Data Load Complete!")
    print(data_kpis)
    print(data_times)

    # worker
    for injection_time in data_times:

        # anomaly detection
        anomaly_kpis = {}
        for kpi_name in data_kpis:
            kpi = data_kpis[kpi_name]
            # TODO
            start_time, degree, is_anomaly = anomaly_detection(injection_time, kpi, args)
            if is_anomaly:
                anomaly_kpis[kpi_name] = get_anomaly(start_time, degree)


if __name__ == "__main__":
    args = parse_args()

    # np.random(args.seed)
    main_worker(args)

    