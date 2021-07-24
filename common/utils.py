import numpy as np
import pickle
import os

def load_single_csv(path, filename):
    import csv
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
        'timestamp': np.array(timestamp),
        'value': np.array(value),
        # 'time_and_val': time_and_val,
    }
    return record


def load_kpis_from_csv(path, cnt=2):
    '''return a dict kpi data'''
    file_list = os.listdir(path)
    data = {}
    if cnt < 0:
        cnt = len(file_list)
    for filename in file_list[0:cnt]:
        print(filename)
        record = load_single_csv(path, filename)
        data[record['kpi_name']] = record
    return data


def load_times_from_csv(filename):
    '''return a list time data'''
    import csv
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