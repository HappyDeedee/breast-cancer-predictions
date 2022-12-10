from dataset_PCR import load_data, Dataset_PCR
import torch
from model import  BinaryClassification
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold 
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


#linear_model.load_state_dict(torch.load("model\save_3_2.150427.model")) #load model

#load data
data_path = 'D:\\ML\\assignment2\\data\\'
file_name = 'trainDataset.xls'
datafile_path = data_path + file_name

all_data = load_data(datafile_path=datafile_path)


# parameters tuning
learningRate = 0.00001
epochs = 2000
batch_mode = 64


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



loss_train_per_model = []
acc_train_per_model = []
acc_eval_per_model = []
loss_eval_per_model = []

def save_model(model, model_name, file_path):
    torch.save(model.state_dict(), file_path + model_name)

def train(train_set, val_set):
    count = 0
    Binary_model = BinaryClassification(input_dimension=117, output_dimension=2)
    Binary_model.to(device)
    mean_evaluatuion_loss = []
    mean_evaluatuion_acc = []
    mean_train_acc = []
    mean_train_loss = []
    #optimizer = torch.optim.Adam(Binary_model.parameters(), lr=learningRate)
    optimizer = torch.optim.SGD(
        Binary_model.parameters(), lr=learningRate, momentum=0.9
    )

    criterion = torch.nn.CrossEntropyLoss()
    
    Binary_model.train()

    def accuracy(pred, gold): 
        pred_copy = pred.clone()
        pred_copy = torch.argmax(pred_copy, 1)
        correct = pred_copy == gold
        return correct.sum().item() / len(correct)

    count += 1
    for epoch in range(epochs):
        
        for i, (data, label_RFS, label_PCR) in enumerate(train_set):  # Converting inputs and labels to Variable
            
            input = data
            labels = label_PCR
            labels = label_PCR.view(-1)
            labels = labels.to(torch.long)
            loss_value = []
            acc_train = []

            input.to(device)
        
            predicted_y = Binary_model(input)
            
            loss = criterion(predicted_y, labels)
            acc_train.append(accuracy(predicted_y, labels))
            optimizer.zero_grad()  # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            loss.backward()  # get gradients w.r.t to parameters
            optimizer.step() # update parameters
 
            # for name, parms in Binary_model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, '--weight', torch.mean(parms.data), ' -->grad_value:', torch.mean(parms.grad))
            loss_value.append(float(loss.detach().numpy()))
            
        print("Loss: ", loss.detach().numpy())
        
        mean_train_loss.append(np.mean(loss_value))
        mean_train_acc.append(np.mean(acc_train))
        writer.add_scalar("acc/train", mean_train_acc[-1], epoch)
        writer.add_scalar("Loss/train", mean_train_loss[-1], epoch)
        
        
        with torch.no_grad():
            eval_loss_batch = []
            eval_acc_batch = []
            Binary_model.eval()
            
            # for j, (data, label_RFS, label_PCR) in enumerate(dataloader_validation):
            for j, (data, label_RFS, label_PCR) in enumerate(val_set):
                input = data[:,:]
                predicted_y = Binary_model(input)
                labels = label_PCR
                labels = label_PCR.view(-1)
                labels = labels.to(torch.long)
                _, pred = torch.max(predicted_y, 1)
                pred = pred.to(torch.float64)
                
                loss = criterion(predicted_y, labels)
                eval_loss_batch.append(loss)
                acc = accuracy(predicted_y, labels)
                eval_acc_batch.append(acc)

        Binary_model.train()
        mean_evaluatuion_loss.append(np.mean(eval_loss_batch))
        mean_evaluatuion_acc.append(np.mean(eval_acc_batch))
        # print(f"Epoch: {epoch}: Evaluation loss: {mean_evaluatuion_loss[-1]}")
        print(f"Epoch: {epoch * count}: Evaluation accuracy: {mean_evaluatuion_acc[-1]}")
        writer.add_scalar("Loss/eval", mean_evaluatuion_loss[-1], epoch * count)
        writer.add_scalar("acc/eval", mean_evaluatuion_acc[-1], epoch * count)
        if epoch % 50 == 0:
            save_model(Binary_model, "model_{}_acc_{}.pt".format(epoch, mean_evaluatuion_acc[-1]), file_path="D:\\ML\\assignment2\\classification_model\\")
    return mean_train_acc, mean_train_loss, mean_evaluatuion_loss, mean_evaluatuion_acc

skf = StratifiedKFold(n_splits=8) 
all_data_ = Dataset_PCR()
all_data_.init(all_data)
x, y1, y2 = all_data_.myGetItem()
#print(x.shape, y1.shape, y2.shape)

x = pd.DataFrame(x)
y2 = pd.DataFrame(y2)

for fold, (train_idx, val_idx) in enumerate(skf.split(x,y2)):  
    train_set,val_set = x.iloc[train_idx], x.iloc[val_idx]
    #print(train_set.shape, val_set.shape)
    train_set_ = Dataset_PCR()
    train_set_.normalisation(train_set)
    #train_set_.dimensionality_reduction(train_set)
    val_set_ = Dataset_PCR()
    val_set_.normalisation(val_set)
    #val_set_.dimensionality_reduction(val_set)
  

    dataloader_train = torch.utils.data.DataLoader(
        train_set_, batch_size=batch_mode, shuffle=True, num_workers=0
    )
    dataloader_validation = torch.utils.data.DataLoader(
        val_set_, batch_size=batch_mode, shuffle=True, num_workers=0
    )

    train_acc, train_loss, eval_loss, eval_acc = train(dataloader_train, dataloader_validation)
    acc_train_per_model.append(train_acc)
    loss_train_per_model.append(train_loss)
    acc_eval_per_model.append(eval_acc)
    loss_eval_per_model.append(eval_loss)

#plot the accuracy for each model to compare
fig, ax = plt.subplots()
for i in range(len(acc_train_per_model)):
    #ax.plot(range(len(acc_train_per_model[i])), acc_train_per_model[i], label=f"model{i}")
    #ax.plot(range(len(loss_eval_per_model[i])), loss_eval_per_model[i], label=f"model{i}")
    ax.plot(range(len(acc_eval_per_model[i])), acc_eval_per_model[i], label=f"model{i}")

plt.title("Accuracy")
ax.set_ylabel("Accuracy")
ax.set_xlabel("Epoch")
plt.legend()
plt.show()

#plot the loss for each model to compare
for i in range(len(loss_train_per_model)):
    fig, ax = plt.subplots()
    ax.plot(range(len(loss_train_per_model[i])), loss_train_per_model[i], label=f"model{i}train loss")
    #ax.plot(range(len(acc_eval_per_model[i])), acc_eval_per_model[i], label=f"model{i}eval loss")
    ax.plot(range(len(loss_eval_per_model[i])), loss_eval_per_model[i], label=f"model{i}eval loss")
plt.title("Loss")
ax.set_ylabel("loss")
ax.set_xlabel("Epoch")
plt.legend()
plt.show()

if __name__ == "__main__":
    #train()
    writer.flush()

    










