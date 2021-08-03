import os
import numpy as np
import pandas as pd
import json
from common.args import parse_args
from common.utils import *
from detection.detector import system_anomaly_detection, service_anomaly_detection
from sklearn.linear_model import LogisticRegression
from builder.pc import pc, gauss_ci_test
from rank.probablistic import page_rank, generate_transfer_matrix


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


def extract_data(anomaly_system_kpis, entity_list):
    num_features = len(entity_list) * 4
    x = []

    # now only have degree features
    # allow to introduce Tc features
    degree_list = []
    for _ in range(len(entity_list)):
        degree_list.append([])
    for kpi_id in anomaly_system_kpis:
        degree = anomaly_system_kpis[kpi_id]
        cmdb_id, _ = decomposition(kpi_id)
        idx = entity_list.index(cmdb_id)

        degree_list[idx].append(degree)
    
    for idx, degrees in enumerate(degree_list):
        if len(degrees) == 0:
            max_d = 0
            min_d = 0
            sum_d = 0
            mean_d = 0
        else:
            max_d = max(degrees)
            min_d = min(degrees)
            sum_d = sum(degrees)
            mean_d = sum_d / len(degrees)
        entity_features = [max_d, min_d, sum_d, mean_d]
        x.extend(entity_features)
    assert len(x) == num_features
    return x


def main_worker(args):
    # init
    assert args.exp_name is not None
    system_kpis, injection_times, service_mrt_data = load_data(args)
    raw_labels = load_labels(args.ground_truth)

    # worker
    processed_features = []
    processed_labels = []
    for injection_time in injection_times:

        print('{1} {0} Start {1}'.format(injection_time, '-'*20))

        # 0. anomaly detection
        print('{0} Anaomaly Detection Start {0}'.format('-'*10))
        anomaly_system_kpis = system_detect(
            injection_time=injection_time,
            system_kpis=system_kpis,
            args=args,
        )

        print('{0} Anaomaly Detection End   {0}'.format('-'*10))
        
        # 0.1 augment/process/extract the data
        # we need to get all features in this part, including:
        #   max_Tc, min_Tc, sum_Tc, mean_Tc;
        #   max_std, min_std, sum_std, mean_std; (this row may be unfeasible)
        #   max_d, min_d, sum_d, mean_d;
        #   ratio; (this row may be unfeasible)
        print('{0} Data augmentation Start {0}'.format('-'*10))
        features = extract_data(anomaly_system_kpis, get_entity_list())
        label = get_entity_list().index(raw_labels[injection_time]["entity"])
        print('{0} Data augmentation End   {0}'.format('-'*10))
        processed_features.append(features)
        processed_labels.append(label)

    # 1. logstic regression to get the entity
    print('{0} Logistic regression Start {0}'.format('-'*10))
    processed_features = np.array(processed_features)
    processed_labels = np.array(processed_labels)
    # divde the training set
    model = LogisticRegression(multi_class='multinomial')
    model.fit(processed_features[:args.train_size], processed_labels[:args.train_size])
    y_hat = model.predict(processed_features[args.train_size:])
    entity_result = []
    entity_list = get_entity_list()
    for label_hat in y_hat:
        entity_result.append(entity_list[label_hat])
    print('{0} Logistic regression End   {0}'.format('-'*10))
    # print(result)

    # 2. process the filtered kpis
    whole_result = []
    for idx, injection_time in enumerate(injection_times[args.train_size:]):
        print('{1} {0} Start {1}'.format(injection_time, '-'*20))
        target_entity = entity_result[idx]
        rank_id = []
        rank_score = []

        # 2.1 redetect
        anomaly_system_kpis = system_detect(
            injection_time=injection_time,
            system_kpis=system_kpis,
            args=args,
        )
        filterd_system_kpis = {}
        for kpi_id in anomaly_system_kpis:
            cmdb_id, _ = decomposition(kpi_id)
            if cmdb_id == target_entity:
                filterd_system_kpis[kpi_id] = anomaly_system_kpis[kpi_id]

        anomaly_service_kpis = service_detect(
            injection_time=injection_time,
            service_mrt_data=service_mrt_data,
            args=args,
        )

        # 2.2 augment data
        data, query_list = merge_system_and_service_kpis(
            timestamp=injection_time,
            window_size=args.pc_window,
            system_kpi_dict=system_kpis,
            query_system_kpis=filterd_system_kpis,
            service_kpi_dict=service_mrt_data,
            query_service_kpis=anomaly_service_kpis,
        )
        print(query_list)

        # 2.3. causal graph learning
        print('{0} Causal graph learning Start {0}'.format('-'*10))
        row_count = sum(1 for row in data)
        corr = pearson_corr(data.values)
        # print(corr)
        cg = pc(
            suffStat={"C": corr, "n": data.values.shape[0]},
            alpha=0.05,
            labels=[str(i) for i in range(row_count)],
            indepTest=gauss_ci_test,
            verbose=True
        )
        print('{0} Causal graph learning End   {0}'.format('-'*10))

        # 2.4 real ranking
        P = generate_transfer_matrix(cg, corr)
        # print(P)
        output_vec = page_rank(P, args.pr_alpha, args.pr_eps, init=-1)
        causal_proba = output_vec[args.pc_service_candidate:] / (output_vec[args.pc_service_candidate:]).sum()
        degree_proba = np.zeros_like(causal_proba)
        for kpi_id in query_list[args.pc_service_candidate:]:
            degree_proba = filterd_system_kpis[kpi_id]
        degree_proba = degree_proba / degree_proba.sum()
        
        rank_id = query_list[args.pc_service_candidate:]
        rank_score = args.rank_gamma * causal_proba + (1 - args.rank_gamma) * degree_proba
        # append the data
        result = gen_json_result(injection_time, rank_id, rank_score)
        whole_result.append(result)

    with open(args.exp_name + '.json', 'w+') as f:
        json_str = json.dumps(whole_result)
        f.write(json_str)

if __name__ == "__main__":
    args = parse_args()

    np.random.seed(args.seed)
    main_worker(args)

    