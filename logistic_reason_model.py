import numpy as np
class LogisticReasonModel:
    def __init__(self):
        self.w = None
        self.b = 0

    def sigmoid(self,z):
        return 1/(1+np.exp(-z))


    def cost_function(self,x,y): #here m is the number of customer_unique_id
        cost_sum = 0
        m,n = x.shape
        for i in range(m):
            z = np.dot(self.w,x[i]) +self.b
            g = self.sigmoid(z)
            cost_sum += -y[i]*np.log(g) -(1-y[i])*np.log(1-g)
        return (1/m)*cost_sum

    def gradient_function(self,x,y, alpha , iterations):
        m,n = x.shape
        self.w = np.zeros(n)
        self.b = 0
        for _ in range(iterations):

            z = np.dot(x,self.w) + self.b
            g = self.sigmoid(z)
            grad_w = (1/m)*(np.dot(x.T,(g-y)))
            grad_b = (1/m)*(np.sum(g-y))


            self.w -= alpha * grad_w
            self.b -= alpha * grad_b

        return self.w,self.b

    def predict(self, x ):
        if self.w is None or self.b is None:
            raise ValueError("Model not trained. Call fit() first.")
        z = np.dot(x, self.w) + self.b
        probs = self.sigmoid(z)
        return np.where(probs >= .62, 1, 0)

    def fit(self,x,y,alpha,iterations):
        self.w, self.b = self.gradient_function(x,y,alpha,iterations)


    def raw_pred(self,x,w,b):
        m,n = x.shape
        g_list = []
        for i in range(m):
            z = np.dot(w,x[i])+ b
            g = self.sigmoid(z)
            g_list.append(g)
        return  g_list


