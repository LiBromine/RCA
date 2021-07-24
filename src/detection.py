import numpy as np
from scipy.stats import norm

def mad(time, value):
    median = np.median(value)
    abs_diff = np.abs(value - median)
    MAD = np.median(abs_diff)


def abs_dev(time, value):
    assert len(time) == len(value)
    if len(time) == 0:
        return None

    early_val = value[:-1]
    val = value[1:]
    abs_diff = np.abs(val - early_val)
    # max_abs_diff = np.max(abs_diff)
    index = np.argmax(abs_diff)
    return time[index]


def guassian_degree(normal_val, query_val):
    mu = np.mean(normal_val)
    sigma = np.std(normal_val)
    o, u = 0, 0
    for val in query_val:
        o += np.log(1 - norm.cdf(val, mu, sigma))
        u += np.log(norm.cdf(val, mu, sigma))
    o = o / (-len(query_val))
    u = u / (-len(query_val))
    return o, u


def anomaly_detection(fail_time, kpi, args):

    # 1. extract a data series 
    # start_time = absolute_derivative(fail_time, args.w1, kpi)
    lower_bound = fail_time - args.w1
    times = kpi['timestamp']
    result = np.where((lower_bound < times) & (times <= fail_time))
    indics = result[0]

    sampled_time = times[indics]
    sampled_val = kpi['value'][indics]
    
    # 2. get change start time
    start_time = abs_dev(sampled_time, sampled_val)
    if start_time is None:
        return None, None, False

    # 3. get change degree
    normal_window_lb = start_time - args.w2
    result = np.where((normal_window_lb < times) & (times <= start_time))
    indics = result[0]
    normal_val = kpi['value'][indics]
    
    new_start_index = indics[-1] + 1
    query_val = kpi['value'][new_start_index:new_start_index+10]
    degree = guassian_degree(normal_val, query_val)
    
    # TODO, a threshold to decide if it is a ananomly KPI
    print(start_time, degree)
    return start_time, degree, True