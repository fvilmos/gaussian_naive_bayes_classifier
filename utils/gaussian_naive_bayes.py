'''
Author: fvilmos
https://github.com/fvilmos
'''
import numpy as np

class GaussianNaiveBayesClassifier():
    # Gaussian Naive Bayes algo implementation
    # input is tabelar data, usually 1st column are the labels, rest the features
    def __init__(self,labels=['M','F'], feature_data_indx=[1,2,3], label_data_indx=0, return_class_label=True):
        self.labels=labels
        self.feature_data_indx=feature_data_indx
        self.label_data_indx = label_data_indx
        self.fmv_list = None
        self.class_prob = None
        self.return_class_label = return_class_label
    
    def fit_data(self, data):
        # tabular data with nxm columbs
        # usually column 0 are the labels, the rest the features
        l_data = []
        for i in range(len(self.labels)):
            col_data = data[data[:,self.label_data_indx] == self.labels[i]]
            l_data.append(col_data)
        
        feature_mean_variance = []
        for i in range(len(self.labels)):
            fmv = self.feature_mean_var(l_data[i],self.feature_data_indx,self.labels[i])
            feature_mean_variance.append(fmv)
        
        self.fmv_list = feature_mean_variance
        
        # class probability
        class_prob = []
        for i in range(len(self.labels)):
            class_prob.append(len(l_data[i][:,0])/len(data[:,0]))

        self.class_prob = class_prob
            
            
    def predict(self,sample):
        # predict based on the fittend data
        prob_feature_list =[]
        posterior_class_list = []
        ret = None
        
        # compute the probabilties based on the features
        for i in range(len(self.labels)):
            prob_feature_list.append([(lambda x: np.log(self.gaussian_dist(x[0],x[1][1],x[1][2])))([sample[j],mv]) for j,mv in enumerate(self.fmv_list[i])])
        
        # compute posterior probabilities
        for i in range(len(self.labels)):
            #Ppost (A|C1) = Pclassprob(A|C1) * P(A|F1)*P(A|F2)*P(A|Fx)
            # ~ Ppost (A|C1) = ln(Pclassprob(A|C1)) + ln(P(A|F1)) + ln(P(A|F2)) + ln(P(A|Fx))
            #posterior_class_list.append(np.log(self.class_prob[i])*np.sum(prob_feature_list[i]))
            posterior_class_list.append(np.log(self.class_prob[i]) + np.sum(prob_feature_list[i]))
        
        #compar and decide based on the bigger probability the sample belonging
        comp_prob = np.argmax(posterior_class_list)
        
        # use labels
        if self.return_class_label == True:
            ret = self.labels[comp_prob]
        else:
            ret = comp_prob
        
        return ret
        
    def gaussian_dist(self,x, miu, var):
        # compute gaussing distribution
        return (1/np.sqrt(2*np.pi*var)*np.e**(-(x-miu)**2/(2*var)))

    def feature_mean_var(self,data,indx=[1,2],id=None):
        # compute mean and vaiance
        f_list=[]
        for f in indx:
            mean_f = np.mean(np.array(data[:,f], dtype=np.float))
            var_f = np.var(np.array(data[:,f], dtype=np.float))
            f_list.append([id,mean_f,var_f])
        return f_list
        
        