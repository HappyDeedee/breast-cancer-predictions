from dataset import load_data, Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

#### load data ####
data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'trainDataset.xls'
datafile_path = data_path + file_name

#### delect missing data ####
missing_value=999
test_dataset = load_data(datafile_path=datafile_path)

for index, row in test_dataset.iterrows():    
    for col in test_dataset.columns:
            if row[col] == missing_value:
                test_dataset.drop(index, inplace=True)

                break
test_dataset = test_dataset.drop(['ID'], axis=1)
test_dataset = test_dataset.reset_index(drop=True)

#### draw a bar chat for each label ####
#sns.countplot(x="pCR (outcome)",data=test_dataset)
sns.scatterplot(x="RelapseFreeSurvival (outcome)", y = 'Age',data=test_dataset)
plt.show()
#draw box chat
dataset = np.array(test_dataset)
fig,ax=plt.subplots(1,figsize=(20,5))
sns.boxplot(data=dataset[:,2:12],ax=ax)
plt.show()
fig,axes=plt.subplots(nrows=24,ncols=5,figsize=(30,80))
fig.subplots_adjust(hspace=.8,wspace=.8)
axes=axes.ravel()

for i,col in enumerate(test_dataset.columns[2:]):
    _=sns.boxplot(y=col,x="pCR (outcome)",data=test_dataset,ax=axes[i])
plt.show()

#### draw feature heatmap ####
corrMatt = test_dataset.corr()
mask = np.zeros_like(corrMatt)
mask[np.triu_indices_from(mask)] = True
plt.title('Breast Cancer Feature Correlation')
cmap = sns.diverging_palette(260, 10, as_cmap=True)
sns.heatmap(corrMatt, vmax=1.2, square=False, cmap=cmap,  fmt='.2g', linewidths=1)
plt.show()


#### check the value range and outliers of each feature ####
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
outliers = []
sns.boxplot(data=scaler.fit_transform(test_dataset))
plt.show()



#### substitute the value range and outliers of each feature ####
def detect_outliers(data, threshold=3.5):
    outliers_index =[]
    mean = np.mean(data)
    std = np.std(data)
    for y in data:
        z_score = (y - mean)/std
        if np.abs(z_score) > threshold:
            outliers_index.append(y.index)
    return outliers_index


def substitute_outliers(data, index):
    upper_bound = data.mean() + 3 * data.std()
    lower_bound = data.mean() - 3 * data.std()
    for i in index:
        if data[i] > upper_bound:
            data[i] = upper_bound
        elif data[i] < lower_bound:
            data[i] = lower_bound
    return data
        
















    

