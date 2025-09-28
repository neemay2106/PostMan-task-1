from logistic_reason_model import LogisticReasonModel
import numpy as np

class CrossValidation:
    def __init__(self):
        self.fold = None

    def split(self,x,y,alpha, iterations):
        self.fold = 10
        parameter = []
        F1 = []
        part_x = np.array_split(x,self.fold)
        part_y = np.array_split(y,self.fold)
        model = LogisticReasonModel()
        for i in range(len(part_x)-1):
            new_list_x = np.vstack(part_x[:i] + part_x[i + 2:])
            new_list_y = np.hstack(part_y[:i] + part_y[i + 2:])
            w, b = model.gradient_function(new_list_x, new_list_y, alpha, iterations)
            parameter.append((w, b))
            test_x = np.vstack([part_x[i], part_x[i + 1]])
            test_y = np.hstack([part_y[i], part_y[i + 1]])
            pred = model.predict(test_x[i], w, b)

            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for j in range(len(pred)):
                if pred[j] == test_y[j] and pred[j] == 1:
                    tp += 1
                elif pred[j] == 1 and test_y[j] == 0:
                    fp += 1
                elif pred[j] == 0 and test_y[j] == 1:
                    fn += 1
                else:
                    tn += 1



            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0

            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0

            f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
            F1.append(f1)

        no_index = F1.index(max(F1))
        w_get,b_get = parameter[no_index]
        return w_get,b_get