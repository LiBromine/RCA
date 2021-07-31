import os
import numpy as np
import pandas as pd
from common.args import parse_args
from common.utils import *
from detection.detector import system_anomaly_detection, service_anomaly_detection
# from trainer.detection import system_anomaly_detection
from builder.pc import pc, gauss_ci_test
from ranker.probablistic import page_rank

def load_data(args):
    '''
    return data format: 
        injection_times: [timestamp0, timestamp1];
        system_kpis: {
            kpi_id: {
                'kpi_name': xxx,
                'cmdb_id': xxx,
                'times': [xx, xx, ..., xx],
                'values': [xx, xx, ..., xx],
            },
            ...
        }
        service_mrt_data: [
            {
                'times': [],
                'mrts': [],
            }, ...
        ]
    '''
    print("Data Load Begin.")
    system_kpis = load_system_kpis_from_csv(args.data_folder, cnt=2)
    injection_times = load_times_from_csv(args.injection_times)
    service_mrt_data = load_service_kpis(args.service_kpi)
    print("Data Load Complete!")
    return system_kpis, injection_times, service_mrt_data


def main_worker(args):
    # init
    system_kpis, injection_times, service_mrt_data = load_data(args)

    # worker
    for injection_time in injection_times:

        # 0. anomaly detection
        print('{1} {0} Begin {1}'.format(injection_time, '-'*20))
        anomaly_system_kpis = {}
        for kpi_id in system_kpis:
            kpi = system_kpis[kpi_id]
            anamalous, degree = system_anomaly_detection(fail_time=injection_time, kpi=kpi, args=args)
            if anamalous:
                anomaly_system_kpis[kpi_id] = degree
                print('{0}: {1}'.format(kpi_id, degree))
        print('At this time, the number of system ananomly is {0}'.format(len(anomaly_system_kpis)))

        anomaly_service_kpis = {}
        for index, data in enumerate(service_mrt_data):
            anamalous, degree = service_anomaly_detection(fail_time=injection_time, mrts=data["mrts"], timestamps=data["times"], args=args)
            if anamalous:
                anomaly_service_kpis[index] = degree
                print('ServiceTest {0}: {1}'.format(index, degree))
        print('At this time, the number of service ananomly is {0}'.format(len(anomaly_service_kpis)))
        return

        # 0.1 augment the data
        # TODO


        # 1. causal graph learning
        # TODO, completely unsolved 
        labels = anomaly_kpis.keys()
        for kpi_id in anomaly_kpis:
            pass

        row_count = sum(1 for row in data)
        cg = pc(
            suffStat = {"C": data.corr().values, "n": data.values.shape[0]},
            alpha = 0.05,
            labels = [str(i) for i in range(row_count)],
            indepTest = gauss_ci_test,
            verbose = True
        )

        # 2. ranking
        # TODO, to construct transition matrix
        P = []
        output_vec = page_rank(P, args.pr_alpha, args.pr_eps, init=-1)  
        print(output_vec)   


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    main_worker(args)

    