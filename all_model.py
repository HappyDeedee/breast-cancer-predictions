from dataset_PCR import load_data, Dataset_PCR
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split







##### load data ####
data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'trainDataset.xls'
datafile_path = data_path + file_name

all_data = load_data(datafile_path=datafile_path)
all_data_ = Dataset_PCR()
all_data_.init(all_data)
train_data, label_RFS, label_PCR = all_data_.myGetItem()
train_data_ = Dataset_PCR()
train_data_.normalisation(train_data)
train_data, label_RFS, label_PCR = train_data_.myGetItem()


train_dataset, validation_dataset, train_label, validation_label = train_test_split(train_data, label_PCR, test_size=0.2, 
random_state=1, stratify=label_PCR)


from sklearn.metrics import accuracy_score

svm=SVC()
svm.fit(X=train_dataset,y=train_label)
predictions=svm.predict(validation_dataset)
print(accuracy_score(validation_label,predictions))
print(confusion_matrix(validation_label,predictions))
print(classification_report(validation_label,predictions)) 




skf = StratifiedKFold(n_splits=5, random_state=None,shuffle=False)
rfc = RandomForestClassifier(n_estimators=87, max_depth=14, max_features=10, random_state=73)
#rfc = RandomForestClassifier(n_estimators=125, max_depth=7, min_samples_split=60, random_state=10, min_samples_leaf=20)
rfc_accuracy = cross_val_score(rfc,train_data,label_PCR.ravel(),cv=skf).mean()
print('accuracy:', rfc_accuracy)