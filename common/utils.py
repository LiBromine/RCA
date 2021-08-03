import numpy as np
import os
import csv
import tqdm

def load_single_csv(path, filename):
    filename = os.path.join(path, filename)

    kpi_name = ''
    cmdb_id = ''
    timestamp = []
    value = []
    time_and_val = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for index, row in enumerate(spamreader):
            if index > 0:
                time_and_val.append((int(row[1]), float(row[4])))
            if index == 1:
                kpi_name = row[3]
                cmdb_id = row[2]
    time_and_val = sorted(time_and_val)
    for item in time_and_val:
        timestamp.append(item[0]) # time
        value.append(item[1]) # value
    record = {
        'kpi_name': kpi_name,
        'cmdb_id': cmdb_id,
        'times': np.array(timestamp),
        'values': np.array(value),
    }
    return record


def load_service_kpis(filename):
    '''returns a list of dict data of service kpi(mrt) 
    keys:
        time: [],
        mrt: [],
    '''
    tc_pos = -1
    mrt_pos = -2
    time_pos = 1
    id_start_pos = 11
    max_rank = 11
    val_list = []
    for _ in range(max_rank):
        val_list.append([])

    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for index, row in enumerate(spamreader):
            if index > 0:
                rank = int(row[tc_pos][id_start_pos:]) - 1
                val_list[rank].append([int(row[time_pos]), float(row[mrt_pos])])
    for rank in range(max_rank):
        timestamps = list()
        mrts = list()
        val_list[rank] = sorted(val_list[rank])
        for pair in val_list[rank]:
            timestamps.append(pair[0]) # time
            mrts.append(pair[1]) # mrt value
        val_list[rank] = {
            "times": np.array(timestamps),
            "values": np.array(mrts),
        }

    return val_list


def load_system_kpis_from_csv(path, cnt=2):
    '''return a dict kpi data
    keys:
        kpi_id: {
            cmdb_id: "",
            kpi_name: "",
            timestamp: [],
            value: [],
        }
    '''
    file_list = os.listdir(path)
    # ===
    # file_list = [
    #     'Tomcat02##OSLinux-CPU_CPU-1_SingleCpuidle.csv',
    #     'Tomcat04##OSLinux-OSLinux_NETWORK_NETWORK_TCP-FIN-WAIT.csv',
    # ]
    # ===
    data = {}
    if cnt < 0:
        cnt = len(file_list)
    for filename in tqdm.tqdm(file_list[0:cnt]):
        print(filename)
        record = load_single_csv(path, filename)
        key = record['cmdb_id'] + '##' + record['kpi_name']
        data[key] = record
    return data


def load_times_from_csv(filename):
    '''return a list time data'''
    timestamp = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for index, row in enumerate(spamreader):
            if index > 0:
                timestamp.append(int(row[0]))
    return np.array(timestamp)


def get_anomaly(start_time, degree):
    return {
        'start_time': start_time,
        'degree': degree,
    }


def get_diff(a):
    return a[1:] - a[:-1]


def merge_system_and_service_kpis(timestamp, window_size, system_kpi_dict, query_system_kpis, service_kpi_dict, query_service_kpis):
    """
    Params:
        timestamp: a query timestamp
        window_size: sampled data number
        system_kpi_dict: a dict data containing all system kpis
        query_system_kpis: some system kpis which are desired to be merged
        service_kpi_dict: a list/dict data containing all list kpis
        query_service_kpis: some service kpis which are desired to be merged
    =============
    Returns:
        data: a pandas dataframe containing all merged data
    """

    query_service_list = list(query_service_kpis.keys())
    if len(query_service_list) == 0:
        only_system_kpis = True
    else: 
        only_system_kpis = False
    query_system_list = list(query_system_kpis.keys())
    query_list = query_service_list + query_system_list

    anchor_key = query_list[0]
    print("Anchor key: {0}".format(anchor_key))
    if not only_system_kpis:
        anchor = service_kpi_dict[anchor_key]
    else:
        anchor = system_kpi_dict[anchor_key]
    anchor_times = anchor["times"]

    interval = 60
    ret_data = []
    # hot_index = 0
    lb = timestamp - interval * 40
    ub = timestamp + interval * 10
    hot_time = lb - interval
    legal_points_num = 0
    loop_exec_num = 0
    while(True):
        # if legal_points_num >= window_size:
        #     break
        # if hot_index >= len(anchor_times) or hot_index < 0:
        #     print("[WARN] Anchor times exhasuted..., has {0} legal data points, stop loop.".format(len(ret_data)))
        #     break
        hot_time += interval
        if hot_time > ub:
            break

        loop_exec_num += 1
        # hot_time = anchor_times[hot_index]
        # hot_index += 1
        is_legal = True
        for key in query_list: # check whether the data of this time point exist in all metrics
            if key in query_service_list:
                data = service_kpi_dict[key]
            else:
                data = system_kpi_dict[key]
            # Initially, we want to the time matched precisely, but it is unfeasible
            # Therefore, we take an approximate strategy using 'data_slice'
            data_slice = data["times"][((hot_time - 2 * interval) <= data["times"]) & (data["times"] <= hot_time)]
            if data_slice.size == 0:
                print("hot_time {0} does not exist in the data slice of key: {1}".format(hot_time, key))
                is_legal = False
                break
        if not is_legal:
            continue
        
        one_time_data = []
        for key in query_list:
            if key in query_service_list:
                data = service_kpi_dict[key]
            else:
                data = system_kpi_dict[key]
            elem = data["values"][((hot_time - 2 * interval) <= data["times"]) & (data["times"] <= hot_time)]
            one_time_data.append(elem[-1]) # approximate the target value using the value in nearest time 
        assert(len(one_time_data) == len(query_list))
        ret_data.append(one_time_data)
        legal_points_num += 1

        
    ret_data = np.array(ret_data)
    import pandas as pd
    return pd.DataFrame(ret_data, columns=query_list), query_list


def pearson_corr(data, eps=1e-3):
    """
    Params:
        data: np.ndarray: (n, k), n is sample number, k is class number
        eps: avoid zero sigma
    =============
    Returns:
        C: np.ndarray: (k, k)
    """
    data = data.T
    k = data.shape[0]
    cov = np.cov(data)
    std = np.std(data, axis=-1)

    # print(data)
    # print(cov)
    # print(std)
    corr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if std[i] == 0 or std[j] == 0:
                corr[i, j] = 0
            else:
                corr[i, j] = cov[i, j] / ((std[i]) * (std[j]))
            # print(i, j, cov[i, j], std[i], std[j], corr[i, j])
            corr[i, j] = np.clip(corr[i, j], -0.9999, 0.9999)
    for i in range(k):
        corr[i, i] = 1.0 
    return corr


def decomposition(kpi_id):
    l = kpi_id.split('##')
    cmdb_id = l[0]
    kpi_name = l[1]
    return cmdb_id, kpi_name


def gen_json_result(injection_time, rank_id, rank_score):
    l = []
    assert len(rank_id) == len(rank_score)
    for i in range(len(rank_id)):
        l.append((rank_score[i], rank_id[i]))
    l = sorted(l)
    l.reverse()

    print(l)
    id_list = []
    for _, kpi_id in l:
        cmdb_id, kpi_name = decomposition(kpi_id)
        id_list.append([cmdb_id, kpi_name])

    return [int(injection_time), id_list]