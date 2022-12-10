
from model import  BinaryClassification
import torch
from dataset import load_data
from sklearn.preprocessing import StandardScaler
import pandas as pd

data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'test1.xls'
datafile_path = data_path + file_name

ID = []
all_data = load_data(datafile_path=datafile_path)
ID = all_data['ID'].copy()
all_data = all_data.drop(['ID'], axis=1)
scaler = StandardScaler()
test_data = scaler.fit_transform(all_data)

model = BinaryClassification(input_dimension=117, output_dimension=2)
model.load_state_dict(torch.load('classification_model\model_1300_acc_0.7708333333333334.pt'))
model.eval()
with torch.no_grad():
    test_data = torch.from_numpy(test_data).double()
    output = model(test_data)
    _, y_pred = torch.max(output.data, 1)
    
y_pred = y_pred.numpy()
y_pred = y_pred.reshape(-1)

result = pd.DataFrame({'ID': ID, 'predicted PCR outcome': y_pred})
result.to_csv('D:\\ML\\assignment2\\result\\predictedPCR_result.csv', index=False)
print('Predictions are closed')



