import numpy as np
import pickle
import os
import csv

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
                # timestamp.append(int(row[1]))
                # value.append(float(row[4]))
                time_and_val.append((int(row[1]), float(row[4])))
            if index == 1:
                kpi_name = row[3]
                cmdb_id = row[2]
    time_and_val = sorted(time_and_val)
    for item in time_and_val:
        timestamp.append(item[0])
        value.append(item[1])
    record = {
        'kpi_name': kpi_name,
        'cmdb_id': cmdb_id,
        'times': np.array(timestamp),
        'values': np.array(value),
        # 'time_and_val': time_and_val,
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
            "mrts": np.array(mrts),
        }

    # row = 0
    # for rank in range(max_rank):
    #     row += len(val_list[rank]["time"])
    #     print(len(val_list[rank]["time"]))
    #     print(len(val_list[rank]["mrt"]))
    # print('--- row : {}'.format(row))
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
    for filename in file_list[0:cnt]:
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