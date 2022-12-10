import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl



device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#check device

#### ElasticLinear model ####
class ElasticLinear(pl.LightningModule):
    def __init__(
        self, loss_fn, n_inputs: int = 1, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05,
    ):
        super().__init__()
        # self.hparams = hparams
        self.learning_rate = learning_rate
        self.n_inputs = n_inputs
        self.loss_fn = loss_fn
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(self.n_inputs, 1).to(torch.float64)
        self.train_log = []
        self.val_log = []
        

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()
        
        return self.l2_lambda * l2_norm

    def training_step(self, batch, batch_idx):
        x, y1, y2 = batch
        y1 = y1.squeeze(0)
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y1) + self.l1_reg() + self.l2_reg()
        self.log("train_loss", loss)
        self.train_log.append(loss.detach().numpy())
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y1, y2 = batch
        y_hat = self(x)
        y1 = y1.squeeze(0)
        loss = self.loss_fn(y_hat, y1)
        self.log("val_loss", loss)
        self.val_log.append(loss.detach().numpy())
        
               
        
        return loss


#### logistics regression ####
class BinaryClassification(nn.Module):
    def __init__(self, input_dimension, output_dimension=2):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dimension, out_features=256).double()
        self.linear2 = torch.nn.Linear(256, 512).to(torch.float64)
        self.linear3 = torch.nn.Linear(512, 256).to(torch.float64)
        self.linear4 = torch.nn.Linear(256, 128).to(torch.float64)
        self.linear5 = torch.nn.Linear(128, 64).to(torch.float64)
        self.linear6 = torch.nn.Linear(64, 32).to(torch.float64)
        self.linear7 = torch.nn.Linear(32, 16).to(torch.float64)
        self.linear8 = torch.nn.Linear(16, output_dimension).to(torch.float64)
        self.apply(self._init_weights)
    def forward(self, x):
        output = self.linear1(x)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear2(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear3(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear4(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear5(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear6(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear7(output)
        output = torch.nn.functional.leaky_relu(output)
        output = self.linear8(output)
        output = torch.nn.functional.softmax(output, dim=1)
        return output

        
    
    def _init_weights(self, m):
        if type(m) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

