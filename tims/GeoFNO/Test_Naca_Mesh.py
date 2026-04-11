
import torch.nn.functional as F
import matplotlib.pyplot as plt
from timeit import default_timer
from utilities3 import *
from Adam import Adam
import torch
from naca_geofno import SpectralConv2d, FNO2d

torch.serialization.add_safe_globals([FNO2d, SpectralConv2d])
################################################################
# configs
################################################################
PATH = "."
INPUT_X = PATH+'/data/naca/NACA_Cylinder_X.npy'
INPUT_Y = PATH+'/data/naca/NACA_Cylinder_Y.npy'
OUTPUT_Sigma = PATH+'/data/naca/NACA_Cylinder_Q.npy'


model_path = "./model/naca_plain_model_500"


model = torch.load(model_path, weights_only=False)
################################################################
# load data and data normalization
################################################################
inputX = np.load(INPUT_X)
inputX = torch.tensor(inputX, dtype=torch.float)
inputY = np.load(INPUT_Y)
inputY = torch.tensor(inputY, dtype=torch.float)
input = torch.stack([inputX, inputY], dim=-1)

output = np.load(OUTPUT_Sigma)[:, 4]
output = torch.tensor(output, dtype=torch.float)
print(input.shape, output.shape)

ntrain = 1000
ntest = 200
r1 = 1
r2 = 1
s1 = int(((221 - 1) / r1) + 1)
s2 = int(((51 - 1) / r2) + 1)

batch_size = 20

x_train = input[:ntrain, ::r1, ::r2][:, :s1, :s2]
y_train = output[:ntrain, ::r1, ::r2][:, :s1, :s2]
x_test = input[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
y_test = output[ntrain:ntrain+ntest, ::r1, ::r2][:, :s1, :s2]
x_train = x_train.reshape(ntrain, s1, s2, 2)
x_test = x_test.reshape(ntest, s1, s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size,
                                          shuffle=False)
test_loader2 = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1,
                                          shuffle=False)

for i in range(5):
    x, y = next(iter(test_loader))


    ind = -1
    X = x[ind, :, :, 0].squeeze().detach().cpu().numpy()
    Y = x[ind, :, :, 1].squeeze().detach().cpu().numpy()
    truth = y[ind].squeeze().detach().cpu().numpy()

    out = model(x.to('cuda')).cpu().detach().numpy()
    pred = out[ind].squeeze()
    nx = 40//r1
    ny = 20//r2
    X_small = X[nx:-nx, :ny]
    Y_small = Y[nx:-nx, :ny]
    truth_small = truth[nx:-nx, :ny]
    pred_small = pred[nx:-nx, :ny]

    fig, ax = plt.subplots(nrows=3, ncols=2,  figsize=(16, 16))
    ax[0,0].pcolormesh(X, Y, truth, shading='gouraud')
    ax[1,0].pcolormesh(X, Y, pred, shading='gouraud')
    ax[2,0].pcolormesh(X, Y, pred-truth, shading='gouraud')
    ax[0,1].pcolormesh(X_small, Y_small, truth_small, shading='gouraud')
    ax[1,1].pcolormesh(X_small, Y_small, pred_small, shading='gouraud')
    ax[2,1].pcolormesh(X_small, Y_small, np.abs(pred_small-truth_small), shading='gouraud')
    plt.show()

