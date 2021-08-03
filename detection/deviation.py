
import numpy as np
import pandas as pd
import tqdm
from scipy.stats import norm

class Dev:
    """
    A anomaly detection object based on deviation
    """
    def __init__(self, method='abs'):
        """
        Constructor

	    Parameters
	    ----------
	    method: computation method
	
	    Returns
	    ----------
    	abs_dev/median abs_dev object
        """
        methods = ['abs', 'mad']
        if method in methods:
            self.method = method
        else:
            print('This method (%s) is not supported' % type(method))
            raise NotImplementedError

    def fit(self, data):
        """
        Import data to Dev object
        
        Parameters
	    ----------
	    data : list, numpy.array or pandas.Series
		    initial batch to calibrate the algorithm
	
        """
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, pd.Series):
            self.data = data.values
        else:
            print('This data format (%s) is not supported' % type(data))
            raise NotImplementedError
    
    @staticmethod
    def _mad(series, sigma_min=1e-4, MAD_min=1e-3, k=3):
        """
        Params:
        series: time series: np.ndarray
        sigma_min: min sigma to avoid sigma near to 0
        k: a threshold setting
        ======================
        Returns:
        dict:
            keys: 'degree', 'alarms'
        """
        median = np.median(series)
        abs_diff = np.abs(series - median)
        MAD = np.median(abs_diff) + MAD_min
        s = (series - median) / MAD
        mu = np.mean(s)
        sigma = np.std(s) + sigma_min
        alarms = []
        degree = []
        record = [] # for log and check error
        for i, val in enumerate(s):
            if (abs(val - mu) >= k * sigma):
                alarms.append(i)
                degree.append(val)
            record.append(val)

        return {'degree': degree, 'alarms': alarms}

    @staticmethod
    def _abs_dev(series, sigma_min=1e-4, k=3):
        """
        Params:
        series: time series: np.ndarray
        sigma_min: min sigma to avoid sigma near to 0
        k: a threshold setting
        ======================
        Returns:
        dict:
            keys: 'max_alarm_index', 'abs_diff', 'alarms'
        """
        early_data = series[:-1]
        newer_data = series[1:]
        abs_diff = np.abs(newer_data - early_data)
        mu = np.mean(abs_diff)
        sigma = np.std(abs_diff) + sigma_min
        alarms = []
        degree = []
        record = [] # for log and check error
        max_dev_index = -1
        for i, diff in enumerate(abs_diff):
            
            if (abs(diff - mu) >= k * sigma):
                alarms.append(i)
                degree.append(abs(diff - mu) / sigma)
                
                max_dev_index = i
            record.append(abs(diff - mu) / sigma)
        # print(record)

        return {'max_alarm_index' : max_dev_index, 'degree': degree, 'alarms': alarms}

    def run(self, sigma_min=1e-4, k=3):
        if self.method == 'abs':
            return self._abs_dev(self.data, sigma_min=sigma_min, k=k)
        elif self.method == 'mad':
            return self._mad(self.data, sigma_min=sigma_min, k=k)
        else:
            raise NotImplementedError

