import numpy as np
from detection.deviation import Dev
from detection.spot import biSPOT 

def no_anomaly():
    return False, 0

def anomaly_detection(method, config, testing_values, init_values=None):
    if method == 'spot':
        model = biSPOT
        detector = model()
        # detector.fit(sampled_val, )
        # TODO
    elif method == 'abs':
        model = Dev
        detector = model(method=method)
        detector.fit(testing_values)
        result = detector.run(k=config['k'])
        
        # now we do not introduce the notion of 'starting time'
        anomalous = (len(result['alarms']) > 0)
        if anomalous:
            print(result['degree'])
            degree = max(np.abs(result['degree']))
            return anomalous, degree
        else:
            return no_anomaly()

    elif method == 'mad':
        model = Dev
        detector = model(method=method)
        detector.fit(testing_values)
        result = detector.run(k=config['k'])

        anomalous = (len(result['alarms']) > 0)
        if anomalous:
            degree = max(np.abs(result['degree']))
            return anomalous, degree
        else:
            return no_anomaly()
    else:
        raise NotImplementedError
    

def service_anomaly_detection(fail_time, mrts, timestamps, args):
    """
    Params:
        fail_time: int; a timestamp;
        mrts: list;
        timestamps: list; 
        args: struct data
    ============
    Returns:
        anomalous: bool,
        degree: number
    """

    # 1. extract a data series 
    testing_start_time = fail_time - args.w1
    # testing_times = times[(testing_start_time <= times) & (times <= fail_time)] # intercept
    testing_values = mrts[(testing_start_time <= timestamps) & (timestamps <= fail_time)] # intercept
    if len(testing_values) <= 1:
        return no_anomaly()
    # print(testing_values)

    # 2. detect and get degree
    config = {
        'k': 3
    }
    return anomaly_detection(method=args.ad_method, config=config, testing_values=testing_values, init_values=None)


def system_anomaly_detection(fail_time, kpi, args):
    """
    Params:
        fail_time: int; timestamp;
        kpi: dict data; keys: kpi_name, cmdb_id, times, values
        args: struct data
    =============
    Returns:
        anomalous: bool,
        degree: number
    """

    # 1. extract a data series 
    testing_start_time = fail_time - args.w1
    times = kpi['times']
    # testing_times = times[(testing_start_time <= times) & (times <= fail_time)] # intercept
    testing_values = kpi['values'][(testing_start_time <= times) & (times <= fail_time)] # intercept
    if len(testing_values) <= 1:
        return no_anomaly()
    
    # 2. detect and get degree
    config = {
        'k': 3
    }
    return anomaly_detection(method=args.ad_method, config=config, testing_values=testing_values, init_values=None)

