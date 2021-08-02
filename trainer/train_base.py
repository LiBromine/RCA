import os
import numpy as np
import pandas as pd
from common.args import parse_args
from common.utils import *
from detection.detector import system_anomaly_detection, service_anomaly_detection
from builder.pc import pc, gauss_ci_test
from ranker.probablistic import page_rank

def system_detect(injection_time, system_kpis, args):
    anomaly_system_kpis = {}
    for kpi_id in system_kpis:
        kpi = system_kpis[kpi_id]
        anamalous, degree = system_anomaly_detection(fail_time=injection_time, kpi=kpi, args=args)
        if anamalous:
            anomaly_system_kpis[kpi_id] = degree

    unsorted_list = []
    for key in anomaly_system_kpis:
        unsorted_list.append((anomaly_system_kpis[key], key))
    sorted_list = sorted(unsorted_list)
    chosen = sorted_list[-args.pc_system_candidate:]
    anomaly_system_kpis.clear()
    for degree, key in chosen:
        anomaly_system_kpis[key] = degree
    print('[INFO] At this time, the chosen number of system ananomly is {0}'.format(len(anomaly_system_kpis)))
    print('[INFO] {0}'.format(anomaly_system_kpis))

    return anomaly_system_kpis


def service_detect(injection_time, service_mrt_data, args):
    anomaly_service_kpis = {}
    for index, data in enumerate(service_mrt_data):
        anamalous, degree = service_anomaly_detection(fail_time=injection_time, mrts=data["values"], timestamps=data["times"], args=args)
        if anamalous:
            anomaly_service_kpis[index] = degree
    
    unsorted_list = []
    for key in anomaly_service_kpis:
        unsorted_list.append((anomaly_service_kpis[key], key))
    sorted_list = sorted(unsorted_list)
    chosen = sorted_list[-args.pc_service_candidate:]
    anomaly_service_kpis.clear()
    for degree, key in chosen:
        anomaly_service_kpis[key] = degree
    print('[INFO] At this time, the chosen number of service ananomly is {0}'.format(len(anomaly_service_kpis)))
    print('[INFO] {0}'.format(anomaly_service_kpis))

    return anomaly_service_kpis


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
                'values': [],
            }, ...
        ]
    '''
    print("Data Load Begin.")
    system_kpis = load_system_kpis_from_csv(args.data_folder, cnt=-1)
    injection_times = load_times_from_csv(args.injection_times)
    service_mrt_data = load_service_kpis(args.service_kpi)
    print("Data Load Complete!")
    return system_kpis, injection_times, service_mrt_data


def main_worker(args):
    # init
    system_kpis, injection_times, service_mrt_data = load_data(args)

    # worker
    for injection_time in injection_times:

        print('{1} {0} Start {1}'.format(injection_time, '-'*20))

        # 0. anomaly detection
        print('{0} Anaomaly Detection Start {0}'.format('-'*10))
        anomaly_system_kpis = system_detect(
            injection_time=injection_time,
            system_kpis=system_kpis,
            args=args,
        )

        anomaly_service_kpis = service_detect(
            injection_time=injection_time,
            service_mrt_data=service_mrt_data,
            args=args,
        )
        print('{0} Anaomaly Detection End   {0}'.format('-'*10))
        
        # 0.1 augment the data
        print('{0} Data augmentation Start {0}'.format('-'*10))
        data, query_list = merge_system_and_service_kpis(
            timestamp=injection_time,
            window_size=args.pc_window,
            system_kpi_dict=system_kpis,
            query_system_kpis=anomaly_system_kpis,
            service_kpi_dict=service_mrt_data,
            query_service_kpis=anomaly_service_kpis,
        )
        print('{0} Data augmentation End   {0}'.format('-'*10))

        # 1. causal graph learning
        print('{0} Causal graph learning Start {0}'.format('-'*10))
        row_count = sum(1 for row in data)
        corr = pearson_corr(data.values)
        cg = pc(
            suffStat={"C": corr, "n": data.values.shape[0]},
            alpha=0.05,
            labels=[str(i) for i in range(row_count)],
            indepTest=gauss_ci_test,
            verbose=True
        )
        print('{0} Causal graph learning End   {0}'.format('-'*10))
        return

        # 2. ranking
        # TODO, to construct transition matrix
        P = []
        output_vec = page_rank(P, args.pr_alpha, args.pr_eps, init=-1)  
        print(output_vec)   


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    main_worker(args)

    