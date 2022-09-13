import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Models.models import Dense_Model, BiGRU_Model, UnetModel, UnetModel2, Lstm
import torch.optim as optim
import torch.nn as nn
import os
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle
from Utils import *
from scipy.signal import savgol_filter
seed = 1
np.random.seed(seed)

gpu = '4'
os.environ["CUDA_VISIBLE_DEVICES"]=gpu

class multiClassHingeLoss(nn.Module):
    def __init__(self, p=1, margin=1, weight=None, size_average=True):
        super(multiClassHingeLoss, self).__init__()
        self.p=p
        self.margin=margin
        self.weight=weight
        self.size_average=size_average

    # define feed-forward		
    def forward(self, output, y):
        output_y=output[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()].view(-1,1)

        # output - output(y) + output(i)
        loss=output-output_y+self.margin

        # remove i=y items
        loss[torch.arange(0,y.size()[0]).long().cuda(),y.data.cuda()]=0
		# apply max function
        loss[loss<0]=0
        
		# apply power p function
        if(self.p!=1):
            loss=torch.pow(loss,self.p)

        # add weight
        if(self.weight is not None):
            loss=loss*self.weight

        # sum up
        loss=torch.sum(loss)

        if(self.size_average):
            loss/=output.size()[0]

        return loss


def Train(logname = '3Dconv', epochs=1000, batch_size=100, early_stop = 50):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    input_datasets = np.load('./lip_data_AB.npy', allow_pickle=True)  
    input_datasets_b2 = np.load('./lip_data_B2.npy', allow_pickle=True)
    input_datasets_b3 = np.load('./lip_data_B3_600.npy', allow_pickle=True)
    input_datasets_a6 = np.load('./lip_data_a6_600.npy', allow_pickle=True)

    input_datasets = np.concatenate((input_datasets,input_datasets_b2,input_datasets_b3,input_datasets_a6),axis = 0)
    
    total_dataset_num = int(input_datasets.shape[0]/100)  # 22개 set
    # print('total dataset num: ',total_dataset_num)
    input_datasets = savgol_filter(input_datasets, 51, 3)
    for a in range(total_dataset_num*100):
        for b in range(8):
            row = input_datasets[a, b, :]
            row = normalize(row)
            input_datasets[a, b, :] = row

    
    ##5-fold example###
    #ttest = [0,1,12,22,30,55,56,57,36,37,38,70,71,72,84,85,86,99,100,101]
   # ttest = [2,3,13,15,23,31,58,59,60,40,41,42,73,74,75,87,88,89,102,103]
    ttest = [5,6,16,17,24,25,32,61,62,63,43,44,45,76,77,78,91,92,104,106]
    # ttest = [7,9,18,19,26,27,33,64,65,66,46,47,48,79,80,93,94,95,107,108]
    #ttest = [10,11,20,21,28,29,34,67,68,69,49,50,51,81,82,96,97,98,109,110]
    
    ttrain = [i for i in range(total_dataset_num)]#A:0~11, A':12~21,A'':22~29 A''''
    for j in ttest:
        ttrain.remove(j)

    
    print('Test  :',ttest)
    print('Train :',ttrain)
    ##############################################

    test_X = np.empty((0, 8, 600))
    test_Y = np.empty((0,))
    train_X = np.empty((0, 8, 600))
    train_Y = np.empty((0,))
    temp4 = []
    temp5 = []

    words_10 = [i for i in range(100)]
   
    for i in ttest:
        temp = input_datasets[i*100:(i+1)*100]
        temp4 = []
        temp5 = []
        for j in words_10:
            temp4.append(temp[j])
        temp = temp4
        test_X = np.append(test_X, temp, axis=0)
        temp5 = [i for i in range(len(words_10))]
        test_Y = np.append(test_Y, temp5, axis=0)
        
    temp4 = []
    temp5 = []

    for i in ttrain:
        temp4 = []
        temp5 = []
        temp = input_datasets[i*100:(i+1)*100]
        for j in words_10:
            temp4.append(temp[j])
        temp = temp4
        train_X = np.append(train_X, temp, axis=0)
        temp5 = [i for i in range(len(words_10))]
        train_Y = np.append(train_Y, temp5, axis=0)
    
    print('%'*50)

    for k in range(1):
        model = UnetModel2().to(device)
        save_path = './ver2/1'

        print('%'*100)
        os.makedirs(save_path, exist_ok=True)
        train_loss_list = list()
        val_loss_list  = list()
        train_acc_list = list()
        val_acc_list   = list()

        hard_criterion = nn.CrossEntropyLoss()
        svm_criterion = multiClassHingeLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))

        early_stop_number = 0
        old_val_acc = 0
        for epoch in range(epochs):
            # print('epoch: ',epoch)
            train_loss = 0.0
            train_correct = 0.0
            train_total = 0.0
            model.train()
            
            for i in tqdm(range(len(train_X)//batch_size)):
                inputs, labels = torch.from_numpy(train_X[i*batch_size:(i+1)*batch_size]), torch.from_numpy(train_Y[i*batch_size:(i+1)*batch_size])

                inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)
        
                inputs = inputs.view(inputs.shape[0], 1, 4, 2, -1)
                
                optimizer.zero_grad()

                outputs, features = model(inputs)

                hard_loss = hard_criterion(outputs, labels)
                loss = hard_loss
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()        
         
                
            model.eval()
            val_loss = 0.0
            val_correct = 0.0
            val_total = 0.0
            confusion_mat = np.zeros((100, 100))
                
            for i in tqdm(range(len(test_X)//batch_size)):
                inputs, labels = torch.from_numpy(test_X[i*batch_size:(i+1)*batch_size]).to(device), torch.from_numpy(test_Y[i*batch_size:(i+1)*batch_size]).to(device)
                inputs = torch.tensor(inputs, dtype=torch.float32, device=device)
                labels = torch.tensor(labels, dtype=torch.long, device=device)
                
                inputs = inputs.view(inputs.shape[0], 1, 4, 2, -1)
                
                outputs, features = model(inputs)
                
                loss = hard_criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                for y in range(batch_size):
                    confusion_mat[labels[y]][predicted[y]] += 1
                
                
                # TSNE FEATURES
                if i == 0:
                    total_features = features.detach().cpu().numpy()
                    total_labels = labels.cpu().numpy()
                else:
                    total_features = np.concatenate((total_features, features.detach().cpu().numpy()))
                    total_labels = np.concatenate((total_labels, labels.cpu().numpy()))
                
            
            print('{}/{} loss: {} acc: {}'.format(epoch, epochs, train_loss / train_total, train_correct / train_total))
            print('{}/{} validation loss: {} acc: {}'.format(epoch, epochs, val_loss / val_total, val_correct/val_total))
            print('*'*100)
            
            color_name = 'Reds'
            confusion_matrix_plot(confusion_mat, list(range(100)), 'emg_'+color_name+'.jpg', cmap=color_name)


            if old_val_acc < val_correct:
                old_val_acc = val_correct

                train_acc = train_correct / train_total
                train_acc *= 100
                train_acc = round(train_acc)

                val_acc = val_correct / val_total
                val_acc *= 100
                val_acc = round(val_acc)

                torch.save(model.state_dict(), os.path.join(save_path, 'weight_epoch{}_{}%_{}%.pt'.format(epoch, train_acc, val_acc)))
                early_stop_number = 0
            else:
                early_stop_number += 1
                if early_stop_number == early_stop:
                    print("Early Stop!")
                    break
            train_loss_list.append(train_loss / train_total)
            val_loss_list.append(val_loss / val_total)
            train_acc_list.append(train_correct/ train_total)
            val_acc_list.append(val_correct / val_total)

            torch.save({
                'epoch'     :   [i+1 for i in range(epoch)],
                'train_loss':   train_loss_list,
                'val_loss'  :   val_loss_list,
                'train_acc' :   train_acc_list,
                'val_acc'   :   val_acc_list
                }, os.path.join(save_path, 'log_{}.pt'.format(logname)))

    w = val_acc_list.index(max(val_acc_list))

    return total_features, total_labels


if __name__=='__main__':

    Train() 
    
