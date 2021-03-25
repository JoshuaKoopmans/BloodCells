import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import random
from skimage import img_as_float
from skimage.feature import peak_local_max
from datetime import datetime
import os
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bullet_perc', help=' percentage bullet (between 0 and 1).', default=1)
parser.add_argument('--round_perc', help=' percentage round (between 0 and 1).', default=1)
args = parser.parse_args()

if os.environ.get("PREFIX") is None:
    prefix = "/mnt/cellstorage/"
else:
    prefix = os.environ.get("PREFIX")
torch.manual_seed(0)
model_id = datetime.now().strftime("%d%m%Y-%H%M%S%p")
model_dir = "{}model_runs/{}/".format(prefix, model_id)
if not os.path.isdir("{}model_runs/".format(prefix)):
    os.mkdir("{}model_runs/".format(prefix))
os.mkdir(model_dir)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=5, padding=28)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, dilation=3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, dilation=3)
        self.bn3 = nn.BatchNorm2d(64)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=5)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=5)
        self.softmax = nn.Softmax()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        x = self.bn1(self.pool(self.relu(self.conv1(x))))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))
        x = self.upsample(x)
        y1 = self.conv4(x)
        y1 = self.softmax(y1)
        y2 = self.conv5(x)
        y2 = F.sigmoid(y2)
        return y1, y2


x_bullet, x_round = torch.load(prefix + "bullet_cells_data.pt").float() / 255., torch.load(
    prefix + "round_cells_data.pt").float() / 255.
y_bullet_gauss, y_round_guass = torch.load(prefix + "bullet_cells_labels_gauss.pt").float() / 255., torch.load(
    prefix + "round_cells_labels_gauss.pt").float() / 255.
y_bullet, y_round = torch.load(prefix + "bullet_cells_labels.pt").float() / 255., torch.load(
    prefix + "round_cells_labels.pt").float() / 255.

bullet_percentage, round_percentage = int(len(y_bullet) * float(args.bullet_perc)), int(len(y_round) * float(args.round_perc))

X = torch.cat((x_bullet[:bullet_percentage], x_round[:round_percentage]), 0)
y_gauss = torch.cat((y_bullet_gauss[:bullet_percentage], y_round_guass[:round_percentage]), 0)
y = torch.cat((y_bullet[:bullet_percentage], y_round[:round_percentage]), 0)
y = torch.cat([torch.cat(((1 - i), i), 0).unsqueeze(0) for i in y], 0)
# y_gauss = torch.cat([torch.cat(((1 - i), i), 0).unsqueeze(0) for i in y_gauss], 0)
## Only cell
# X = torch.cat([i[0].unsqueeze(0) for i in X], 0).unsqueeze(1)
#
# # print(X[0].squeeze(0).numpy().shape)
# # print(y.shape)
# import cv2
# cv2.imwrite('a.png',X[0].squeeze(0).numpy()*255)
# cv2.imwrite('b.png',y[0].squeeze(0).numpy()*255)
# #######
xtest = X[-10:]
ytest = y[-10:]
# import cv2
# cv2.imwrite('a.png',X[0][0].numpy()*255)
# cv2.imwrite('b.png',X[0][1].numpy()*255)

# X = X[:100]
# y = y[:100]
import numpy as np


def train(X, y, mini_batch_size: int):
    model = Net()
    with open(model_dir + "model_info.txt", "w") as f:
        print(model, file=f, end="\n")
        print("Training data amounts:\nBullet: {} - Round: {}".format(str(bullet_percentage), str(round_percentage)), file=f)
        f.close()
    # mlt = MultiTaskLoss2(n_loss=2)
    optimizer = torch.optim.Adam(model.parameters())
    best_loss = np.inf
    count = 0
    done = False
    epoch = 0

    while not done:
        gc.collect()
        positions = torch.randperm(len(X))
        for mb in range(0, len(X), mini_batch_size):
            x_minib = X[positions[mb:mb + mini_batch_size]]
            y_minib = y[positions[mb:mb + mini_batch_size]]
            y_gauss_minib = y_gauss[positions[mb:mb + mini_batch_size]]
            y_pred, y_pred_gauss = model(x_minib)
            #weight = y_minib * 0.98 + (1 - y_minib) * 0.02  # same size as y
            loss_train = F.binary_cross_entropy(y_pred, y_minib.float())
            loss_train_gauss = F.binary_cross_entropy(y_pred_gauss, y_gauss_minib.float())
            total_loss = loss_train + loss_train_gauss
            # total_loss = mlt([loss_train, loss_train_gauss])
            # Maakt dit verschil? (met log of zonder log) loss_train = F.log(F.binary_cross_entropy(y_pred, y_minib.float()))
            print(loss_train)

            if loss_train.item() < best_loss:
                best_loss = loss_train.item()
            if loss_train.item() >= best_loss:
                count += 1

            total_loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 0:
                model.eval()
                evaluate(model, epoch)
                torch.save(model.state_dict(), prefix + "convnet.pt")
                model.train()
            epoch += 1


test = {"002": [torch.tensor(cv2.imread(prefix + "resources/002_2.5kfps/002_2.5kfps_98.png", 0)) / 255,
                torch.tensor(cv2.imread(prefix + "median_real_002.png", 0)) / 255],
        "K1": [torch.tensor(cv2.imread(prefix + "resources/K1_001_20201105/K1_001_20201105_168.png",
                                       0)) / 255,
               torch.tensor(cv2.imread(prefix + "median_real_K1.png", 0)) / 255],
        "NF": [
            torch.tensor(cv2.imread(prefix + "resources/NF135_002_20201105/NF135_002_20201105_157.png",
                                    0)) / 255,
            torch.tensor(cv2.imread(prefix + "median_real_NF.png", 0)) / 255]}


def evaluate(model, num):
    t1 = torch.cat((test["002"][0].unsqueeze(0), test["002"][1].unsqueeze(0)), 0)
    t2 = torch.cat((test["K1"][0].unsqueeze(0), test["K1"][1].unsqueeze(0)), 0)
    t3 = torch.cat((test["NF"][0].unsqueeze(0), test["NF"][1].unsqueeze(0)), 0)
    xtest = torch.cat((t1, t2, t3), 2).unsqueeze(0)
    # xtest = list(torch.cat((v[0].unsqueeze(0), v[1].unsqueeze(0)), 0) for k,v in test.items()) #torch.cat((xtest.unsqueeze(0), background.unsqueeze(0)), 0).unsqueeze(0)).float()
    # xtest = [torch.tensor(x) for x in xtest]
    # print(xtest.shape)
    # ytest = torch.tensor(cv2.imread("/home/joshua/Stage 2021/scripts/median_real.png", 0))
    # ytest = ytest.unsqueeze(0).unsqueeze(0)
    # ytest = torch.cat([torch.cat(((1 - i), i), 0).unsqueeze(0) for i in ytest], 0)
    with torch.no_grad():
        pred, pred_gauss = model(xtest)
        a = torch.cat(([i[0] for i in pred]), 1)
        e = torch.cat(([i[0] for i in pred_gauss]), 1)
        e = [pred_gauss[0][0]]
        f = torch.cat(([get_coordinates(i, xtest[0][0]) for i in e]), 1)
        # print(pred.min(), pred.max())
        # b = torch.cat(([i[0] for i in ytest]), 1)
        d = torch.cat(([i[0] for i in xtest]), 1)
        # c = torch.cat((d, e, f,  a), 0)
        c = torch.cat((f, pred_gauss[0][0], pred[0][0]), 0)
        cv2.imwrite(model_dir + 'evaluate_{}.png'.format(num), c.detach().numpy() * 255)


def get_coordinates(img, img2):
    img = img.detach().numpy()
    img2 = img2.detach().numpy()
    print(img.shape)
    coordinates = peak_local_max(img, threshold_abs=.3, min_distance=10)
    print(coordinates.shape)
    for coord in coordinates:
        #     print(tuple(coord))
        img2 = cv2.circle(img2, center=(coord[1], coord[0]), radius=30, thickness=2, color=255)
    return torch.tensor(img2)


train(X, y, 64)

# prediction = net(X[:100])
#
# print(prediction.shape)
#
# import os
# import numpy as np
# directory = "/home/joshua/Stage 2021/resources/K1_001_20201105/" #NF135_002_20201105, 002_2.5kfps
# images = []
# for filename in os.listdir(directory)[:1000]:
#
#     path = directory + filename
#     images.append(np.array(cv2.imread(path, 0)))
# images = np.array(images)
#
# median = np.median(images, 0)
# mean = np.mean(images, 0)
# cv2.imwrite("median_real_K1.png", median)
# cv2.imwrite("mean_real_K1.png", mean)
# # median_background = cv2.resize(cv2.imread("/home/joshua/Stage 2021/scripts/median_real_K1.png", 0), (400,400))
# # cv2.imwrite("median_scaled.png", median_background)
