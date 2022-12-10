import xgboost as xgb
import pickle
from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from scipy.sparse import csr_matrix,hstack
from sklearn.model_selection import KFold,train_test_split
from xgboost import XGBRegressor
from dataset_PCR import load_data, Dataset_PCR
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'trainDataset.xls'
datafile_path = data_path + file_name




all_data = load_data(datafile_path=datafile_path)
all_data_ = Dataset_PCR()
all_data_.init(all_data)
train_data, label_RFS, label_PCR = all_data_.myGetItem()
# train_data_ = Dataset_PCR()
# train_data_.normalisation(train_data)
#train_data, label_RFS, label_PCR = train_data_.myGetItem()
train_data = train_data.drop(['pCR (outcome)'], axis=1)
train_data = train_data.drop(['RelapseFreeSurvival (outcome)'], axis=1)
# scaler = StandardScaler()
# train_data = scaler.fit_transform(train_data)
normal_train_data =(train_data-train_data.mean())/train_data.std()
print(type(normal_train_data))

train_dataset, validation_dataset, train_label, validation_label = train_test_split(normal_train_data, label_RFS, test_size=0.2, 
random_state=1)


DTrain = xgb.DMatrix(train_data, label=label_RFS)

xgb_params={
    'seed':0,
    'eta':0.1,
    'colsample_bytree':0.5,
    'subsample':0.5,
    'objective':'reg:linear',  
    'max_depth':5,
    'min_child_weight':3
}

          
# instantiate the classifier 
xgb_lr = XGBRegressor(**xgb_params)


# fit the classifier to the training data
xgb_lr.fit(train_dataset, train_label)


bst_cv2=xgb.cv(xgb_params,DTrain,
              num_boost_round=500,
              nfold=3,seed=3,
              metrics= 'mae',
              maximize=False,
              early_stopping_rounds=10,
              verbose_eval=10)
print('CV score:',bst_cv2)


#plot top 10 features
xgb.plot_importance(xgb_lr, max_num_features=20, height=0.5)
xgb_lr.feature_importances_
plt.figure(figsize = (20, 12))
plt.yticks(rotation = 70)
plt.rc('axes',  labelsize=1)

plt.show()