import os
import sys
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import numpy as np
from Eval import Eval
from math import log, exp
import time
import nltk
from imdb import IMDBdata
from collections import Counter
from nltk.corpus import stopwords
from Vocab import Vocab

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.X.shape[1]
        self.count_positive = np.zeros([1,data.X.shape[1]])
        self.count_negative = np.zeros([1,data.X.shape[1]])
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.deno_pos = 0.0
        self.deno_neg = 0.0
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()

        
        self.num_positive_reviews = len(positive_indices)
        self.num_negative_reviews = len(negative_indices)
        
        self.count_positive = csr_matrix.sum(X[np.ix_(positive_indices)], axis=0) #np.ones([1,X.shape[1]])
        self.count_negative = csr_matrix.sum(X[np.ix_(negative_indices)], axis=0)#np.ones([1,X.shape[1]])
        
        self.total_positive_words = csr_matrix.sum(X[np.ix_(positive_indices)])
        self.total_negative_words = csr_matrix.sum(X[np.ix_(negative_indices)])
        
        self.deno_pos = float(self.total_positive_words + self.ALPHA * X.shape[1])
        self.deno_neg = float(self.total_negative_words + self.ALPHA * X.shape[1])

        self.count_positive = (self.count_positive + self.ALPHA) 
        self.count_negative = (self.count_negative + self.ALPHA) 
        
        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X, probThresh, flag):
        #TODO: Implement Naive Bayes Classification
        self.P_positive = log(float(self.num_positive_reviews)) - log(float(self.num_positive_reviews + self.num_negative_reviews))
        self.P_negative = log(float(self.num_negative_reviews)) - log(float(self.num_positive_reviews + self.num_negative_reviews))
        pred_labels = []
        #print(self.P_positive)
        sh = X.shape[0]
               
        for i in range(sh):
            z = X[i].nonzero()
            possum = self.P_positive
            negsum = self.P_negative
            for j in range(len(z[0])):
                # Look at each feature
                r = i  #row index
                c =z[1][j]   #column index
                count = X[r, c]

                prob_pos = log(self.count_positive[0, c]) - log(self.deno_pos)
                possum = possum + count * prob_pos
                
                prob_neg = log(self.count_negative[0, c]) - log(self.deno_neg)
                negsum = negsum + count * prob_neg
                pass
            predicted_prob_positive = exp(possum - self.LogSum(possum, negsum))
            predicted_prob_negative = exp(negsum - self.LogSum(possum, negsum))

            if flag==False:
                if predicted_prob_positive > probThresh:
                    pred_labels.append(1.0)
                else :
                    pred_labels.append(-1.0)

            else :
                if possum > negsum:
                    # Predict positive
                    pred_labels.append(1.0)
                else:
                    # Predict negative
                    pred_labels.append(-1.0)
        
        return pred_labels

    def LogSum(self, logx, logy):   
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)        
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):
                
        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            z = test.X[i].nonzero()
            sum_positive = self.P_positive
            sum_negative = self.P_negative
            predicted_label = 0
            
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                count = test.X[row_index, col_index]
                #print('count::'%count)
                                
                prob_pos = log(self.count_positive[0, col_index])
                sum_positive = sum_positive + count * prob_pos

                prob_neg = log(self.count_negative[0, col_index])
                sum_negative = sum_negative + count * prob_neg
            
            predicted_prob_positive = exp(sum_positive - self.LogSum(sum_positive, sum_negative))
            predicted_prob_negative = exp(sum_negative - self.LogSum(sum_positive, sum_negative))

            if sum_positive > sum_negative:
                predicted_label = 1.0
            else:
                predicted_label = -1.0
                      
            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])

            
    # Evaluate performance on test data 
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X,0,True)
        recall_pos = []
        recall_neg = []
        precision_pos = []
        precision_neg =[]
        ev = Eval(Y_pred, test.Y)
        accy = ev.Accuracy()
        #print(ev.EvalRecall())
        #print(ev.EvalPrecision())
        probThresh = [0.2,0.4,0.6,0.8]
        for i in range(len(probThresh)):
            pred = self.PredictLabel(test.X,probThresh[i],False) 
            ev = Eval(pred, test.Y)
            #ev.EvalRecall()
            #print('Threshold Value %f'%probThresh[i])
            recallP,recallN=ev.EvalRecall()
            recall_pos.append(recallP)
            recall_neg.append(recallN)
            precisionP,precisionN=ev.EvalPrecision()
            precision_pos.append(precisionP)
            precision_neg.append(precisionN)
        plt.pause(0.1)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        fig=plt.figure()
        plt.title('For +1 Label')
        plt.plot(recall_pos,precision_pos,'r')
        fig.savefig('Pos_class.png')
        plt.title('For -1 Label')
        fig1=plt.figure(1)
        plt.plot(recall_neg,precision_neg,'b')
        fig1.savefig('Neg_class.png')
           
        return accy
    #Top 20 Positive and Negative words from vocab.
    def WordWeight(self,directory,testdata):
        pFiles = os.listdir("%s/pos" % directory)
        wordFreq ={}
        wordWt ={}
        #unwanted_chars = ".,-_()"
        stop_words = set(stopwords.words('english'))
        for i in range(len(pFiles)):
            f = pFiles[i]
            lines = ""
            for line in open("%s/pos/%s" % (directory, f), encoding="utf8"):
                lines += line
                wordCounts = Counter([testdata.vocab.GetID(w.lower()) for w in line.split(" ")])
                for (wordId, count) in wordCounts.items():
                    if wordId >= 0:
                        word = testdata.vocab.GetWord(wordId)
                        if word not in stop_words:
                            #word = word.strip(unwanted_chars)
                            if word not in wordFreq:
                                wordFreq[word] = 1
                            else :
                                wordFreq[word] += 1

        self.P_positive = exp(self.P_positive)
        self.P_negative = exp(self.P_negative)
       
        for (w,freq) in wordFreq.items():
            wordWt[w]=float((freq/(self.total_positive_words))*self.P_positive)
        Top_Wt=(sorted(wordWt.items(), key=lambda x:-x[1]))[0:21]
        print('Top 20 Positive words')
        print(Top_Wt)

        #for negative words:
        nFiles = os.listdir("%s/neg" % directory)
        wordFreqN ={}
        wordWtN ={}
        unwanted_chars = " (and so on)"
        for i in range(len(nFiles)):
            f = nFiles[i]
            lines = ""
            for line in open("%s/neg/%s" % (directory, f), encoding="utf8"):
                lines += line
                wordCounts = Counter([testdata.vocab.GetID(w.lower()) for w in line.split(" ")])
                #print(wordCounts)
                a=testdata.vocab.GetWord(192)
                #print(a)
                #userinput = input('Enter')
                for (wordId, count) in wordCounts.items():
                    if wordId >= 0:
                        word = testdata.vocab.GetWord(wordId)
                        if word not in stop_words:
                            #word = word.strip(unwanted_chars)
                            if word not in wordFreqN:
                                wordFreqN[word] = 1
                            else :
                                wordFreqN[word] += 1
        self.P_positive = exp(self.P_positive)
        self.P_negative = exp(self.P_negative)
        for (w,freq) in wordFreqN.items():
            wordWtN[w]=float((freq/(self.total_negative_words))*self.P_positive)
        Top_Wt=(sorted(wordWtN.items(), key=lambda x:-x[1]))[0:21]
        print('Top 20 Negative words')
        print(Top_Wt)
                             
        
        return 0

if __name__ == "__main__":
    
    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    #PredictProbValue = nb.PredictProb(testdata, range(10))
    #Weight=nb.WordWeight("%s/test" %sys.argv[1],testdata)
    #print(PredictProbValue)

