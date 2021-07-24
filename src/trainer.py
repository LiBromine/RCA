import os
import numpy as np
from common.args import parse_args
from common.utils import *
from src.detection import anomaly_detection

def load_data(args):
    '''
    data format: 
        times: [timestamp0, timestamp1];
        kpis: {
            kpi_name: {
                'kpi_name': xxx,
                'cmdb_id': xxx,
                'timestamp': [xx, xx, ..., xx],
                'value': [xx, xx, ..., xx],
            },
            ...
        }
    '''
    print("Data Load Begin.")
    data_kpis = load_kpis_from_csv(args.data_folder)
    data_times = load_times_from_csv(args.injection_times)
    print("Data Load Complete!")
    return data_kpis, data_times    

def main_worker(args):
    # init
    data_kpis, data_times = load_data(args) 

    # worker
    for injection_time in data_times:

        # anomaly detection
        anomaly_kpis = {}
        for kpi_name in data_kpis:
            kpi = data_kpis[kpi_name]
            start_time, degree, is_anomaly = anomaly_detection(injection_time, kpi, args)
            if is_anomaly:
                anomaly_kpis[kpi_name] = get_anomaly(start_time, degree)
            
        
        break


if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    main_worker(args)

    