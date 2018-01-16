import numpy as np
import pandas as pd
from pandas import ExcelWriter
import openpyxl

class saveTopologyMap(object):

    def __init__(self, file_name, path):
        #Assign required variables first
        self._file_name = file_name
        self._path = path

    def toCsv(self, TopologyMap_position, weighted_vector):
        print(TopologyMap_position)

