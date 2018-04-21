import numpy as np

class Eval:
    def __init__(self, pred, gold):
        self.pred = pred
        self.gold = gold
        
    def Accuracy(self):
        return np.sum(np.equal(self.pred, self.gold)) / float(len(self.gold))

    def EvalRecall(self):
        prediction = list(self.pred)
        FN=0
        TP=0
        FP=0
        TN=0
        actual = list(self.gold)
        for i in range(len(actual)):
            if prediction[i]== -1 and actual[i]==1:
                FN += 1
            elif prediction[i]== 1 and actual[i]==1:
                TP += 1
            elif prediction[i]== 1 and actual[i]==-1:
                FP += 1
            elif prediction[i]== -1 and actual[i]==-1:
                TN +=1
        
        return float(TP/(TP+FN)), float(TN/(TN+FP))
    
    def EvalPrecision(self):
        prediction = list(self.pred)
        FP=0
        TP=0
        FN=0
        TN=0
        actual = list(self.gold)
        for i in range(len(actual)):
            if prediction[i]== -1 and actual[i]==1:
                FN += 1
            elif prediction[i]== 1 and actual[i]==1:
                TP += 1
            elif prediction[i]== 1 and actual[i]==-1:
                FP += 1
            elif prediction[i]== -1 and actual[i]==-1:
                TN +=1
        return float(TP/(TP+FP)), float(TN/(TN+FN))
    
    
