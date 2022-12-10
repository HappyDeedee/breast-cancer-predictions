from model import ElasticLinear
import torch
import matplotlib.pyplot as plt
from dataset import load_data, Dataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(dirpath="linear_model",
                                      filename="save_{epoch:02d}_{val_loss:.6f}.model",
                                      save_top_k=3,
                                      monitor="val_loss",
                                      every_n_epochs=1,
                                      mode="min")


#### set datafile path ####
data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'trainDataset.xls'
datafile_path = data_path + file_name

#### load data ####
all_data = load_data(datafile_path=datafile_path)
train_data,validation_data = train_test_split(all_data, test_size=117, random_state=117)
train_dataset = Dataset(train_data)
validation_dataset = Dataset(validation_data)

# #### model parameter ####
model = ElasticLinear(  
    #loss_fn=torch.nn.MSELoss(),
    loss_fn=torch.nn.L1Loss(),#MAE
    n_inputs=25,
    l1_lambda=0.05,
    l2_lambda=0.2,
    learning_rate=0.001,
)
# hyper_params = torch.load('linear_model\save_epoch=36_val_loss=20.665835.model.ckpt')
# model.load_state_dict(hyper_params['state_dict'])
batch_size = 32
train_epoch = 2200
#### load model ####
#model.load_from_checkpoint('linear_model\save_epoch=1299_val_loss=21.088690.model.ckpt')


dataloader_train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
dataloader_validation = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, num_workers=0)  


# fit model
trainer = pl.Trainer(max_epochs=train_epoch,log_every_n_steps=64, callbacks=[checkpoint_callback])
trainer.fit(model, dataloader_train, dataloader_validation)




w_model = np.append(
    model.output_layer.bias.detach().numpy()[0],
    model.output_layer.weight.detach().numpy(),
)

def plot_convergence(train_loss):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(train_loss)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train loss')
    
    fig.show()

plot_convergence(model.train_log)

