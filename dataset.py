
import torch.utils.data as data
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler








def load_data(datafile_path:str)->pd.DataFrame:
    # Load data from excel file, and return a dataframe
    data = pd.read_excel(datafile_path, header=0)
    return data






class Dataset(data.Dataset):


    def __init__(self, data):
        self.original_data = data
        self.clean_data = self.delete_missing_data(self.original_data, missing_value=999)
        self.clean_data = self.clean_data.reset_index(drop=True)
        self.clean_data = self.clean_data.drop(['ID'], axis=1)
        for i in range(2, 119):
             self.out_liers_index = self.detect_outliers(self.clean_data.iloc[:, i])
             #print("number of outliers", len(self.out_liers_index))
             self.clean_data.iloc[:, i] = self.substitute_outliers(self.clean_data.iloc[:,i], self.out_liers_index)

        self.clean_data_copy = self.clean_data
        self.clean_data_copy = np.array(self.clean_data_copy)
        self.clean_data = self.feature_normalisation(self.clean_data)
        self.clean_data = self.dimensionality_reduction(self.clean_data)
        self.clean_data = np.array(self.clean_data)

    def __getitem__(self, index):
        
        label_RFS = self.clean_data_copy[index,:][1:2]
        label_PCR= self.clean_data_copy[index,:][0:1]
        return self.clean_data[index], label_RFS, label_PCR
    
    def __len__(self)->int:
        return len(self.clean_data)

    def myGetItem(self):
        label_RFS = self.clean_data_copy[:,1:2]
        label_PCR= self.clean_data_copy[:,0:1]
        return self.clean_data[:], label_RFS, label_PCR
        
    def delete_missing_data(self, data:pd.DataFrame, missing_value=999)->pd.DataFrame:
        # Delete missing data
        for index, row in data.iterrows():
            for col in data.columns:
                if row[col] == missing_value:
                    data.drop(index, inplace=True)
                    break
        return data
    
    def detect_outliers(self, data, threshold=3.5):
        outliers_index =[]
        mean = np.mean(data)
        std = np.std(data)
        for y in range(len(data)):
            z_score = (data[y] - mean)/std
            if np.abs(z_score) > threshold:
                outliers_index.append(y)
        return outliers_index
    
    def substitute_outliers(self, data, index):
        upper_bound = data.mean() + 3 * data.std()#99.7%
        lower_bound = data.mean() - 3 * data.std()
        data_copy = data.copy()
        for i in index:
            if data[i] > upper_bound:
                data_copy.loc[i] = upper_bound
            elif data[i] < lower_bound:
                data_copy.loc[i] = lower_bound
        return data
    
        

    def feature_normalisation(self, clean_data):
        clean_data = clean_data.drop(['pCR (outcome)'], axis=1)
        clean_data = clean_data.drop(['RelapseFreeSurvival (outcome)'], axis=1)
        
        scaler = StandardScaler()
        data_normalisation = scaler.fit_transform(clean_data)
        
        
        return data_normalisation

    def dimensionality_reduction(self, data_normalisation, N_components=25):
         # print(clean_data)
        pca = PCA(n_components=N_components)
        principalComponents = pca.fit_transform(data_normalisation)
          
        print("Pca Contribution value",sum(pca.explained_variance_ratio_))
         
        return principalComponents
         
        











