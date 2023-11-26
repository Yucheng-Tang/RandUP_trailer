import numpy as np
import openpyxl
from .utility import Utility


class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    def data_processing(data_path):
        wb = openpyxl.load_workbook(data_path)
        ws_is = wb['Initial States']
        ws_ob = wb['Obstacles']
        ws_ia = wb['Intersection Areas']

        data_is = Utility.xl_2_numpy(ws_is)
        data_ob = Utility.xl_2_numpy(ws_ob)
        data_ia = Utility.xl_2_numpy(ws_ia)
        distance = Utility.relative_distance(data_is, data_ob)
        inputs = np.concatenate((distance, data_is[:, 3].reshape(data_is.shape[0], 1)), axis=1)
        labels = data_ia
        return inputs, labels
