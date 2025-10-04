import pandas as pd
import numpy as np


class CrossValidation:
    def __init__(self):
        self.fold = None

    def split(self,x,y,model_class,model_para = {},args = (), fit_kwargs = {}, pred_args = ()):
        self.fold = 10
        F1_score = []
        models= []
        #splitting the dataset into flods
        part_x = np.array_split(x,self.fold)
        part_y = np.array_split(y,self.fold)

        for i in range(self.fold):
            #Picking 1 fold for testing and the rest 9 left for training
            x_train = np.vstack([part_x[j] for j in range(self.fold) if j != i])
            y_train = np.hstack([part_y[j] for j in range(self.fold) if j != i])
            x_test = part_x[i]
            y_test = part_y[i]


            #Loading in the model we want to to Cross Validate
            model = model_class(**model_para)
            model.fit(x_train,y_train,*args,**fit_kwargs)
            pred = model.predict(x_test,*pred_args)
            y_test = y_test.to_numpy()




            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for j in range(len(pred)):
                if pred[j] == y_test[j] and pred[j] == 1:
                    tp += 1
                elif pred[j] == 1 and y_test[j] == 0:
                    fp += 1
                elif pred[j] == 0 and y_test[j] == 1:
                    fn += 1
                else:
                    tn += 1
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
            f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0
            F1_score.append(f1)
            models.append(model)
        #Getting index for maximum F1 score
        best_idx = np.argmax(F1_score)


        return models[best_idx], F1_score