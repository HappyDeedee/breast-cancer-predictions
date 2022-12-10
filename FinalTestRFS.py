
from model import ElasticLinear
import torch
from dataset import load_data
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'test1.xls'
datafile_path = data_path + file_name

#### load data ####
ID = []
all_data = load_data(datafile_path=datafile_path)
ID = all_data['ID'].copy()
all_data = all_data.drop(['ID'], axis=1)
scaler = StandardScaler()
test_data = scaler.fit_transform(all_data)
pca = PCA(n_components=25)
test_data = pca.fit_transform(test_data)


model = ElasticLinear(  
    #loss_fn=torch.nn.MSELoss(),
    loss_fn=torch.nn.L1Loss(),#MAE
     n_inputs=25,
    # l1_lambda=0,
    # l2_lambda=0,
    #learning_rate=0.001,
)
hyper_params = torch.load('linear_model\save_epoch=2199_val_loss=24.820040.model.ckpt')
model.load_state_dict(hyper_params['state_dict'])

model.eval()
with torch.no_grad():
    model.to("cpu")
    test_data = torch.from_numpy(test_data).double()
    test_data = test_data.to("cpu")
    y_pred = model(test_data)

y_pred = y_pred.numpy()
y_pred = y_pred.reshape(-1)

#write to csv
df = pd.DataFrame({'ID':ID, ' predicted RFS outcome':y_pred})
df.to_csv('D:\\ML\\assignment2\\result\\predictedRFS_result.csv', index=False)
print('Predictions are closed')
