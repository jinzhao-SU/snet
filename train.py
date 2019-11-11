import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.optim import lr_scheduler
from model import MainNet
from submodel import SubNet
from dataloader import UAVDatasetTuple
from utils import draw_roc_curve, calculate_precision_recall, visualize_sum_testing_result, visualize_sum_training_result
from correlation import Correlation


os.environ["CUDA_VISIBLE_DEVICES"]="0"


def embedding(input, time_stamp=None):
    task_md = torch.zeros([input.shape[0], 100, 100])
    #print("input shape", input.shape)
    for i in range(input.shape[0]):
        x1 = int(input[i][0])
        y1 = int(input[i][1])
        x2 = int(input[i][2])
        y2 = int(input[i][3])
        if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
            continue
        else:
            task_md[i][x1][y1] = 1.00
            task_md[i][x2][y2] = 1.00
    return task_md.view(-1,10000).float()

def train(submodel, model, train_loader, device, optimizer, criterion, epoch, batch_size):
    submodel.eval()
    model.train()
    sum_running_loss = 0.0
    loss_mse = 0.0
    num_images = 0

    for batch_idx, data in enumerate(tqdm(train_loader)):
        optimizer.zero_grad()

        task = data['task'].to(device).float()
        #print("task shape", task.shape)
        init = data['init'].to(device).float()
        #print("init shape", init.shape)
        label = data['label'].to(device).float()
        #model prediction
        sub_prediction = torch.zeros(task.shape[0], task.shape[1], 100, 100, dtype=torch.float).to(device)
        #batch,60,15,4
        for i in range(task.shape[0]):#b 60 15 5
            # print("i shape", task[:,i,:].shape)
            for j in range(task.shape[1]):
                prediction_timestamp = submodel(embedding(task[i,j,:,:].squeeze()).to(device))
            #input 15 4 output 15 100 100
                prediction_timestamp = torch.sum(prediction_timestamp, dim=0)
                sub_prediction[i][j] = prediction_timestamp
                # b 60 .100 100.
        sub_prediction = torch.unsqueeze(sub_prediction,1)
        prediction = model(subx=sub_prediction, mainx=init)
        #loss
        loss_mse = criterion(prediction, label.data)

        # visualize_sum_training_result(init, prediction, sub_prediction, label.data, batch_idx, epoch, batch_size)

        # update the weights within the model
        loss_mse.backward()
        optimizer.step()

        #accumulate loss
        if loss_mse != 0.0:
            sum_running_loss += loss_mse * init.size(0)
        num_images += init.size(0)

        if batch_idx % 50 == 0 or batch_idx == len(train_loader) - 1:
            sum_epoch_loss = sum_running_loss / num_images
            visualize_sum_training_result(init, prediction, sub_prediction, label.data, batch_idx, epoch, batch_size)
            print('\nTraining phase: epoch: {} batch:{} Loss: {:.4f}\n'.format(epoch, batch_idx, sum_epoch_loss))


def val(submodel, model, test_loader, device, criterion, epoch, batch_size):
    submodel.eval()
    model.eval()
    sum_running_loss = 0.0

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            task = data['task'].to(device).float()
            init = data['init'].to(device).float()
            # print("init shape", init.shape)
            label = data['label'].to(device).float()
            # model prediction
            sub_prediction = torch.zeros(task.shape[0], task.shape[1], 100, 100, dtype=torch.float).to(device)
            # batch,60,15,4
            for i in range(task.shape[0]):  #
                # print("i shape", task[:,i,:].shape)
                for j in range(task.shape[1]):
                    prediction_timestamp = submodel(embedding(task[i, j, :, :].squeeze()).to(device))
                    # input 15 4 output 15 100 100
                    prediction_timestamp = torch.sum(prediction_timestamp, dim=0)
                    sub_prediction[i][j] = prediction_timestamp
                    # b 60 .100 100.
            sub_prediction = torch.unsqueeze(sub_prediction, 1)
            prediction = model(subx=sub_prediction, mainx=init)
            # loss
            loss_mse = criterion(prediction, label.data)

            # accumulate loss
            sum_running_loss += loss_mse.item() * init.size(0)

            # visualize the sum testing result
            visualize_sum_testing_result(init, prediction, sub_prediction, label.data, batch_idx, epoch, batch_size)

    sum_running_loss = sum_running_loss / len(test_loader.dataset)

    prediction_output = prediction.cpu().detach().numpy()
    label_output = label.cpu().detach().numpy()

    print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(epoch, sum_running_loss))
    return sum_running_loss, prediction_output, label_output

def save_model(checkpoint_dir, model_checkpoint_name, model):
    model_save_path = '{}/{}'.format(checkpoint_dir, model_checkpoint_name)
    print('save model to: \n{}'.format(model_save_path))
    torch.save(model.state_dict(), model_save_path)


def main():
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="data path", required=True, type=str)
    parser.add_argument("--init_path", help="init path", required=True, type=str)
    parser.add_argument("--label_path", help="label path", required=True, type=str)
    parser.add_argument("--lr", help="learning rate", required=True, type=float)
    parser.add_argument("--momentum", help="momentum", required=True, type=float)
    parser.add_argument("--weight_decay", help="weight decay", required=True, type=float)
    parser.add_argument("--batch_size", help="batch size", required=True, type=int)
    parser.add_argument("--num_epochs", help="num_epochs", required=True, type=int)
    parser.add_argument("--split_ratio", help="training/testing split ratio", required=True, type=float)
    parser.add_argument("--checkpoint_dir", help="checkpoint_dir", required=True, type=str)
    parser.add_argument("--model_checkpoint_name", help="model checkpoint name", required=True, type=str)
    parser.add_argument("--load_from_checkpoint", type=str)
    parser.add_argument("--eval_only", dest='eval_only', action='store_true')
    args, unknown = parser.parse_known_args()

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)

    device = torch.device("cuda")

    all_dataset = UAVDatasetTuple(task_path=args.data_path, init_path=args.init_path, label_path=args.label_path)
    # positive_ratio, negative_ratio = all_dataset.get_class_count()
    # weight = torch.FloatTensor((positive_ratio, negative_ratio))
    train_size = int(args.split_ratio * len(all_dataset))
    test_size = len(all_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(all_dataset, [train_size, test_size])
    print("Total image tuples for train: ", len(train_dataset))
    print("Total image tuples for test: ", len(test_dataset))

    print("\nLet's use", torch.cuda.device_count(), "GPUs!\n")
    model_ft = MainNet()
    submodel = SubNet()
    model_ft = nn.DataParallel(model_ft)
    submodel = nn.DataParallel(submodel)

    criterion  = nn.MSELoss(reduction='sum')

    if args.load_from_checkpoint:
        chkpt_model_path = args.load_from_checkpoint
        print("Loading ", chkpt_model_path)
        submodel.load_state_dict(torch.load(chkpt_model_path, map_location=device))
        #submodel = torch.load(chkpt_model_path)


    model_ft = model_ft.to(device)
    submodel = submodel.to(device)

    # Observe that all parameters are being optimized
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Decay LR by a factor of 0.1
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=30,drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=30, drop_last=True)

    cor = Correlation()

    # if args.eval_only:
    #     loss, prediction_output, label_output = val(model_ft, test_loader, device, criterion, 0)
    #     print('\nTesting phase: epoch: {} Loss: {:.4f}\n'.format(0, loss))
    #     cor.corrcoef(prediction_output, label_output, ".", "correlation_test.png")
    #     return True

    best_loss = np.inf
    coef = 0.0
    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 80)
        exp_lr_scheduler.step()
        train(submodel, model_ft, train_loader, device, optimizer_ft, criterion, epoch, args.batch_size)
        loss, prediction_output, label_output = val(submodel,model_ft, test_loader, device, criterion, epoch, args.batch_size)
        if loss < best_loss:
            save_model(checkpoint_dir=args.checkpoint_dir,
                       model_checkpoint_name=args.model_checkpoint_name + '_' + str(loss),
                       model=model_ft)
            best_loss = loss
        cor_path = args.checkpoint_dir.replace("check_point", "testing_result")
        cor_path = os.path.join(cor_path, "epoch_" + str(epoch))
        coef = cor.corrcoef(prediction_output, label_output, cor_path, "correlation_{0}.png".format(epoch))
        print('correlation coefficient : {0}\n'.format(coef))

if __name__ == '__main__':
    main()