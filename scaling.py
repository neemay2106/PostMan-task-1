import math
import numpy as np

class Scaling:
    def __init__(self,x):
        self.x = x

    def scaling_function(self):
        n = self.x.shape[1]
        for i in range(n):
            log_trans = [math.log(j, 10) for j in self.x.iloc[:, i].tolist()]
            x_max = max(log_trans)
            x_min = min(log_trans)
            d = x_max - x_min
            for x_ in log_trans:
                x_scaled = (x_ - x_min) / (x_max - x_min)