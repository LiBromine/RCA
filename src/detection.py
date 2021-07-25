import numpy as np
from scipy.stats import norm
from common.utils import get_diff
from sklearn.neighbors import KernelDensity


def mad(time, value):
    median = np.median(value)
    abs_diff = np.abs(value - median)
    MAD = max(np.median(abs_diff), 1e-7)
    metric = (value - median) / MAD
    index = np.argmax(metric)
    return time[index]


def abs_dev(time, value):
    assert len(time) == len(value)
    if len(time) == 0:
        return None
    elif len(time) == 1:
        return time[0]

    early_val = value[:-1]
    val = value[1:]
    abs_diff = np.abs(val - early_val)
    # max_abs_diff = np.max(abs_diff)
    index = np.argmax(abs_diff)
    return time[index]


def diff_guassian_degree(normal_val, query_val):
    normal_diff = get_diff(normal_val)
    query_diff = get_diff(query_val)
    return guassian_degree(normal_diff, query_diff)


def guassian_degree(normal_val, query_val):
    mu = np.mean(normal_val)
    sigma = max(np.std(normal_val), 1e-7) # avoid sigma equal to zero
    o, u = 0, 0
    # print('Mu: {}, Sigma: {}'.format(mu, sigma))
    for val in query_val:
        # print("Val: {0}, CDF: {1}".format(val, norm.cdf(val, mu, sigma)))
        input_o = 1 - norm.cdf(val, mu, sigma)
        input_o = 1e-11 if input_o == 0.0 else input_o
        input_u = norm.cdf(val, mu, sigma)
        input_u = 1e-11 if input_u == 0.0 else input_u
        o += np.log(input_o)
        u += np.log(input_u)
    o = o / (-len(query_val))
    u = u / (-len(query_val))
    return o, u


def kde_guassian_degree(normal_val, query_val):
    sample = normal_val.reshape(len(normal_val), 1)
    print(sample)
    model = KernelDensity(bandwidth=1, kernel='gaussian')
    model.fit(sample)
    values = query_val.reshape((len(query_val), 1))
    print(values)
    log_prob = model.score_samples(values)
    print(log_prob)
    degree = log_prob.sum() / (-len(query_val))
    return degree


def anomaly_detection(fail_time, kpi, args):

    # 1. extract a data series 
    # start_time = absolute_derivative(fail_time, args.w1, kpi)
    lower_bound = fail_time - args.w1
    times = kpi['timestamp']
    result = np.where((lower_bound <= times) & (times <= fail_time))
    indics = result[0].tolist()
    if len(indics) != 0:
        extra_index = indics[-1] + 1 # try to add one more index to the indics list
        if extra_index < len(times):
            indics.append(extra_index)

    sampled_time = times[indics]
    sampled_val = kpi['value'][indics]
    
    # 2. get change start time
    start_time = abs_dev(sampled_time, sampled_val)
    if start_time is None or start_time > fail_time:
        print('No start time or late start time')
        return None, None, False
    print('Start Time is {0}'.format(start_time))

    # 3. get change degree
    normal_window_lb = start_time - args.w2
    result = np.where((normal_window_lb < times) & (times <= start_time))
    indics = result[0]
    normal_val = kpi['value'][indics]
    
    new_start_index = indics[-1] + 1 # Q: why not plus 1? A: for difference use
    query_val = kpi['value'][new_start_index:new_start_index+args.j_window]
    if len(query_val) < 1 or len(normal_val) <= 1:
        return None, None, False
    degree = guassian_degree(normal_val, query_val)
    
    # 4. compare with a threshold to decide if it is a ananomly KPI
    print(start_time, degree)
    if max(degree) < args.ananomly_th:
        return None, None, False
    
    return start_time, degree, True