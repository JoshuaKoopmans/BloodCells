import cv2
import torch.optim

from Models import *
from skimage.feature import peak_local_max
from datetime import datetime
import os
import sys
import gc
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--bullet_perc', help=' percentage bullet (between 0 and 1).', default=1)
parser.add_argument('--round_perc', help=' percentage round (between 0 and 1).', default=1)
parser.add_argument("--type", help="Choose which model type to run.", choices=["normal", "experimental", "tracking"],
                    default="normal")
parser.add_argument("--stop_epoch", help="Indicate at which epoch the training should be stopped.", type=int,
                    default=-1)
parser.add_argument("--name", help="Indicate folder name where model results are saved.", default="")
parser.add_argument("--checkpoint_file", help="Indicate path of existing model.", default="")
parser.add_argument("--new_data", help="Indicate usage of new data.", type=bool, default=False)
args = parser.parse_args()

torch.manual_seed(0)
EXHAUST_EPOCH = 7000

model_type = str(args.type).lower()
stop_epoch = int(args.stop_epoch) if 0 < int(args.stop_epoch) <= EXHAUST_EPOCH else -1
if os.environ.get("PREFIX") is None:
    prefix = "/mnt/cellstorage/"
else:
    prefix = os.environ.get("PREFIX")

model_id = str(datetime.now().strftime("%d%m%Y-%H%M%S%p")) + "_" + str(args.name)
model_dir = "{}model_runs/{}/".format(prefix, model_id)
if not os.path.isdir("{}model_runs/".format(prefix)):
    os.mkdir("{}model_runs/".format(prefix))
os.mkdir(model_dir)
os.mkdir(model_dir + "models/")

models = {"normal": Net(), "experimental": NetExperiment(), "tracking": NetTracking()}
new_data_file_postfix = "_new" if args.new_data and args.checkpoint_file != "" else ""
x_bullet, x_round = torch.load(
    prefix + "bullet_cells{}_data.pt".format(new_data_file_postfix)).float() / 255., torch.load(
    prefix + "round_cells{}_data.pt".format(new_data_file_postfix)).float() / 255.
y_bullet_gauss, y_round_guass = torch.load(
    prefix + "bullet_cells{}_labels_gauss.pt".format(new_data_file_postfix)).float() / 255., torch.load(
    prefix + "round_cells{}_labels_gauss.pt".format(new_data_file_postfix)).float() / 255.
y_bullet, y_round = torch.load(
    prefix + "bullet_cells{}_labels.pt".format(new_data_file_postfix)).float() / 255., torch.load(
    prefix + "round_cells{}_labels.pt".format(new_data_file_postfix)).float() / 255.

bullet_percentage, round_percentage = int(len(y_bullet) * float(args.bullet_perc)), int(
    len(y_round) * float(args.round_perc))

X = torch.cat((x_bullet[:bullet_percentage], x_round[:round_percentage]), 0)
y_gauss = torch.cat((y_bullet_gauss[:bullet_percentage], y_round_guass[:round_percentage]), 0)
y = torch.cat((y_bullet[:bullet_percentage], y_round[:round_percentage]), 0)
y = torch.cat([torch.cat(((1 - i), i), 0).unsqueeze(0) for i in y], 0)


def train(X, y, mini_batch_size: int):
    model = models[model_type]
    with open(model_dir + "model_info.txt", "w") as f:
        print(model, file=f, end="\n")
        print("Training data amounts:\nBullet: {} - Round: {}".format(str(bullet_percentage), str(round_percentage)),
              file=f)
        f.close()

    optimizer = torch.optim.Adam(model.parameters())
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
            loss_train = F.binary_cross_entropy(y_pred, y_minib.float())
            loss_train_gauss = F.binary_cross_entropy(y_pred_gauss, y_gauss_minib.float())
            total_loss = loss_train + loss_train_gauss

            print("Epoch: {}\nLoss: {}".format(epoch, loss_train.item()), end="\n")

            total_loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 0:
                model.eval()
                evaluate(model, epoch)
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state,
                           model_dir + "models/" + "state_b{}_r{}_{}.pt".format(str(bullet_percentage),
                                                                                str(round_percentage),
                                                                                str(epoch)))
                model.train()
            if stop_epoch > 0:
                if epoch == stop_epoch:
                    model.eval()
                    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state,
                               model_dir + "models/" + "{}_model_epoch_{}.pt".format(model_type, str(stop_epoch)))
                    sys.exit(0)
            if epoch == EXHAUST_EPOCH:
                sys.exit(0)
            epoch += 1


def continue_train(X, y, mini_batch_size: int, checkpoint: str):
    model = models[model_type]
    optimizer = torch.optim.Adam(model.parameters())
    model, optimizer, epoch = load_checkpoint(model=model, optimizer=optimizer, filename=checkpoint)
    model.train()
    with open(model_dir + "model_info.txt", "w") as f:
        print(model, file=f, end="\n")
        print("\nCONTINUING TRAINING\nTraining data amounts:\nBullet: {} - Round: {}".format(str(bullet_percentage),
                                                                                             str(round_percentage)),
              file=f)
        f.close()
    done = False

    while not done:
        gc.collect()
        positions = torch.randperm(len(X))
        for mb in range(0, len(X), mini_batch_size):
            x_minib = X[positions[mb:mb + mini_batch_size]]
            y_minib = y[positions[mb:mb + mini_batch_size]]
            y_gauss_minib = y_gauss[positions[mb:mb + mini_batch_size]]
            y_pred, y_pred_gauss = model(x_minib)
            loss_train = F.binary_cross_entropy(y_pred, y_minib.float())
            loss_train_gauss = F.binary_cross_entropy(y_pred_gauss, y_gauss_minib.float())
            total_loss = loss_train + loss_train_gauss

            print("Epoch: {}\nLoss: {}".format(epoch, loss_train.item()), end="\n")

            total_loss.backward()

            torch.nn.utils.clip_grad_value_(model.parameters(), 1)
            optimizer.step()
            optimizer.zero_grad()
            if epoch % 10 == 0:
                model.eval()
                evaluate(model, epoch)
                state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()}
                torch.save(state,
                           model_dir + "models/" + "model_b{}_r{}_{}.pt".format(str(bullet_percentage),
                                                                                str(round_percentage),
                                                                                str(epoch)))
                model.train()
            if stop_epoch > 0:
                if epoch == stop_epoch:
                    model.eval()
                    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
                             'optimizer': optimizer.state_dict()}
                    torch.save(state,
                               model_dir + "models/" + "{}_model_epoch_{}.pt".format(model_type, str(stop_epoch)))
                    sys.exit(0)
            if epoch == EXHAUST_EPOCH:
                sys.exit(0)
            epoch += 1


def load_checkpoint(model, optimizer, filename):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


if not args.checkpoint_file:
    test = {"002": [torch.tensor(cv2.imread(prefix + "resources/002_2.5kfps/002_2.5kfps_98.png", 0)) / 255,
                    torch.tensor(cv2.imread(prefix + "median_real_002.png", 0)) / 255],
            "K1": [torch.tensor(cv2.imread(prefix + "resources/K1_001_20201105/K1_001_20201105_168.png",
                                           0)) / 255,
                   torch.tensor(cv2.imread(prefix + "median_real_K1.png", 0)) / 255],
            "NF": [
                torch.tensor(cv2.imread(prefix + "resources/NF135_002_20201105/NF135_002_20201105_157.png",
                                        0)) / 255,
                torch.tensor(cv2.imread(prefix + "median_real_NF.png", 0)) / 255]}
else:
    test = {"002": [torch.tensor(cv2.imread(prefix + "resources/002_2.5kfps/002_2.5kfps_222.png", 0)) / 255,
                    torch.tensor(cv2.imread(prefix + "median_real_002.png", 0)) / 255],
            "K1": [torch.tensor(cv2.imread(prefix + "resources/K1_001_20201105/K1_001_20201105_2.png",
                                           0)) / 255,
                   torch.tensor(cv2.imread(prefix + "median_real_K1.png", 0)) / 255],
            "NF": [
                torch.tensor(cv2.imread(prefix + "resources/NF135_002_20201105/NF135_002_20201105_280.png",
                                        0)) / 255,
                torch.tensor(cv2.imread(prefix + "median_real_NF.png", 0)) / 255]}


def evaluate(model, num):
    test_image_1 = torch.cat((test["002"][0].unsqueeze(0), test["002"][1].unsqueeze(0)), 0)
    test_image_2 = torch.cat((test["K1"][0].unsqueeze(0), test["K1"][1].unsqueeze(0)), 0)
    test_image_3 = torch.cat((test["NF"][0].unsqueeze(0), test["NF"][1].unsqueeze(0)), 0)
    test_images = torch.cat((test_image_1, test_image_2, test_image_3), 2).unsqueeze(0)

    with torch.no_grad():
        pred, pred_gauss = model(test_images)
        prediction_side_by_side = torch.cat(([i[0] for i in pred]), 1)
        gauss_prediction_side_by_side = torch.cat(([i[0] for i in pred_gauss]), 1)
        gauss_prediction_side_by_side = [pred_gauss[0][0]]
        cell_coordinates_on_real_data = torch.cat(
            ([get_coordinates(i, test_images[0][0]) for i in gauss_prediction_side_by_side]), 1)

        c = torch.cat((cell_coordinates_on_real_data, pred_gauss[0][0], pred[0][0]), 0)
        cv2.imwrite(model_dir + 'evaluate_{}.png'.format(num), c.detach().numpy() * 255)


def get_coordinates(img, img2):
    img = img.detach().numpy()
    img2 = img2.detach().numpy()
    coordinates = peak_local_max(img, threshold_abs=.3, min_distance=10)
    n_circles_drawn = coordinates.shape[0]
    print("Circles drawn: {}".format(n_circles_drawn), end="\n")
    for coord in coordinates:
        img2 = cv2.circle(img2, center=(coord[1], coord[0]), radius=30, thickness=2, color=255)
    return torch.tensor(img2)


if args.checkpoint_file:
    continue_train(X, y, 64, checkpoint=args.checkpoint_file)
else:
    train(X, y, 64)
