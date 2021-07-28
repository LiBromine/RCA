import os
import numpy as np
import pandas as pd
from common.args import parse_args
from common.utils import *
from trainer.detection import anomaly_detection
from builder.pc import pc, gauss_ci_test
from ranker.probablistic import page_rank

def load_data(args):
    '''
    data format: 
        times: [timestamp0, timestamp1];
        kpis: {
            kpi_id: {
                'kpi_name': xxx,
                'cmdb_id': xxx,
                'timestamp': [xx, xx, ..., xx],
                'value': [xx, xx, ..., xx],
            },
            ...
        }
    '''
    print("Data Load Begin.")
    data_kpis = load_kpis_from_csv(args.data_folder, cnt=-1)
    data_times = load_times_from_csv(args.injection_times)
    print("Data Load Complete!")
    return data_kpis, data_times    


def main_worker(args):
    # init
    data_kpis, data_times = load_data(args) 

    # worker
    for injection_time in data_times:

        # anomaly detection
        print('{1} {0} Begin {1}'.format(injection_time, '-'*20))
        anomaly_kpis = {}
        for kpi_id in data_kpis:
            kpi = data_kpis[kpi_id]
            start_time, degree, is_anomaly = anomaly_detection(injection_time, kpi, args)
            if is_anomaly:
                anomaly_kpis[kpi_id] = get_anomaly(start_time, degree)
                print('{0} Begin'.format(kpi_id))
        print('At this time, the number of ananomly is {0}'.format(len(anomaly_kpis)))
            
        # causal graph learning
        # TODO, completely unsolved 
        data = []
        row_count = sum(1 for row in data)
        cg = pc(
            suffStat = {"C": data.corr().values, "n": data.values.shape[0]},
            alpha = 0.05,
            labels = [str(i) for i in range(row_count)],
            indepTest = gauss_ci_test,
            verbose = True
        )

        # ranking
        # TODO, to construct transition matrix
        P = []
        output_vec = page_rank(P, args.pr_alpha, args.pr_eps, init=-1)  
        print(output_vec)   


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    main_worker(args)

    