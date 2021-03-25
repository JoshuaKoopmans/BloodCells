import sys

import imageio
import numpy as np
import cv2 as cv
from methods import *

# cell = cv.imread("/home/joshua/Stage 2021/scripts/a.png", 0)
# width, height = cell.shape
# print(width)
# cell = cell[25:75, 25:75]
X = torch.cat(
    (torch.load("bullet_cells_data.pt").float() / 255.,
     torch.load("round_cells_data.pt").float() / 255.), 0)

X = X[10:, 0, 20:80, 20:80]
y = torch.cat((torch.load("bullet_cells_labels.pt").float() / 255.,
               torch.load("round_cells_labels.pt").float() / 255.), 0)
y = y[10:, 0, 20:80, 20:80]
# y = y[10:,0,25:75,25:75]

real = (torch.tensor(imageio.imread('real.png')[:, :, 0]) / 255.).unsqueeze(0)

print(X.shape)
print(y.shape)
# gauss2d = cv.imread("/home/joshua/Stage 2021/scripts/label_test.png", 0)/255
import torch
import torch.nn as nn
import torch.nn.functional as F


class HungryFeedForward(nn.Module):
    def __init__(self):
        super(HungryFeedForward, self).__init__()
        self.input_size = 60*60
        self.fc1 = nn.Linear(self.input_size, 500)
        self.bn1 = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 250)
        self.bn2 = nn.BatchNorm1d(250)

        self.fc24 = nn.Linear(250, 500)
        self.bn24 = nn.BatchNorm1d(500)
        self.fc3 = nn.Linear(500, self.input_size)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        # x = F.relu(self.bn22(self.fc22(x)))
        # x = F.relu(self.bn23(self.fc23(x)))
        x = F.relu(self.bn24(self.fc24(x)))
        output = self.fc3(x)
        output = self.sigmoid(output)
        output = output.reshape(shape)
        return output


def train(X, y, mini_batch_size: int):
    model = HungryFeedForward()
    optimizer = torch.optim.Adam(model.parameters())
    best_loss = np.inf
    count = 0
    done = False
    epoch = 0
    while not done:
        positions = torch.randperm(len(X))
        for mb in range(0, len(X), mini_batch_size):
            x_minib = X[positions[mb:mb + mini_batch_size]]
            y_minib = y[positions[mb:mb + mini_batch_size]]

            y_pred = model(x_minib)
            # weight = y_minib * 0.98 + (1 - y_minib) * 0.02  # same size as y
            loss_train = torch.log(F.binary_cross_entropy(y_pred, y_minib.float()))

            # Maakt dit verschil? (met log of zonder log) loss_train = F.log(F.binary_cross_entropy(y_pred, y_minib.float()))
            print(loss_train)

            if loss_train.item() < best_loss:
                best_loss = loss_train.item()
            if loss_train.item() >= best_loss:
                count += 1

            loss_train.backward()

            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 0:
                model.eval()
                print(x_minib.max(), real.max())

                #x_minib = torch.cat((real, x_minib), 0)
                #y_minib = torch.cat((real, y_minib), 0)
                evaluate(model, x_minib[:10], y_minib[:10])
                model.train()
            if count > 5000:
                model.eval()
                torch.save(model.state_dict(), "FF.pt")
                sys.exit(0)
            # if epoch > 2021:
            #     model.eval()
            #     torch.save(model.state_dict(), "FF.pt")
            #     sys.exit(0)
            epoch += 1


def evaluate(model, xtest, ytest):
    print(xtest.shape)
    pred = model(xtest)

    with torch.no_grad():
        pred = model(xtest)
        a = torch.cat(([i for i in pred]), 1)

        # print(pred.min(), pred.max())
        b = torch.cat(([i for i in ytest]), 1)
        d = torch.cat(([i for i in xtest]), 1)
        print(b.shape)
        print(d.shape)
        c = torch.cat((d, a, b), 0)
        cv.imwrite('evalutate_feed.png', c.detach().numpy() * 255)


#train(X, y, 640)

def test():
    x = (torch.tensor(imageio.imread('real.png')[:, :, 0]) / 255.).unsqueeze(0)
    print(x.shape)
    model = HungryFeedForward()
    with torch.no_grad():
        model.load_state_dict(torch.load("FF.pt"))
        model.eval()
        pred = model(x)
        print(pred.shape)
        imageio.imwrite("prediction_real.png", pred[0,:,:].numpy()*255)

#test()

def local_max():
    from scipy import ndimage as ndi
    import matplotlib.pyplot as plt
    from skimage.feature import peak_local_max
    from skimage import data, img_as_float

    im = img_as_float(cv.imread("/home/joshua/Stage 2021/scripts/evalutate_black.png", 0))

    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    #image_max = ndi.maximum_filter(im, size=20, mode='constant')

    # Comparison between image_max and im to find the coordinates of local maxima
    print(im.min(), im.max())
    coordinates = peak_local_max(im, threshold_abs=.3, min_distance=30)
    print(coordinates.shape)
    # display results
    fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()
    ax[0].imshow(im, cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[0].set_title('Original')

    # ax[1].imshow(image_max, cmap=plt.cm.gray)
    # ax[1].axis('off')
    # ax[1].set_title('Maximum filter')

    ax[1].imshow(im, cmap=plt.cm.gray)
    ax[1].autoscale(False)
    ax[1].plot(coordinates[:, 1], coordinates[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('Peak local max')

    fig.tight_layout()

    plt.show()

local_max()