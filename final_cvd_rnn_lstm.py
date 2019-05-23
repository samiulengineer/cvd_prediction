import pandas as pd
import numpy as np
import random
import sklearn
import seaborn as sb

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.autograd import Variable

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, confusion_matrix

import matplotlib.pyplot as plt

from cnn_rnn_model import *

#load data
df = pd.read_csv('/home/samiulengineer/Desktop/sami/data/heart_train.csv')
# df = pd.read_csv('./ok.csv')
# df.dropna(inplace = True)
# df.to_csv('./ok.csv')
data = df.values[:,:-1]
data[data=='?'] = np.nan
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if not pd.isnull(data[i,j]):
            data[i,j] = float(data[i,j])
label = df.values[:,-1].astype('float32')
label[label>0] = 1

#pos and neg example to fill nan
disease_index = np.where(label == 1)[0]
nondisease_index = np.where(label == 0)[0]
disease_data = data[disease_index,:]
nondisease_data = data[nondisease_index,:]

column_num = len(data[0])

new_data = np.zeros((data.shape))
for column in range(column_num):
    pos_data = disease_data[:,column] 
    pos_data = np.delete(pos_data, np.where(pd.isnull(pos_data)), 0)
    pos_avg = np.mean(pos_data)
    pos_avg = round(pos_avg)

    neg_data = nondisease_data[:,column] 
    neg_data = np.delete(neg_data, np.where(pd.isnull(neg_data)), 0)
    neg_avg = np.mean(neg_data)
    neg_avg = round(neg_avg)

    column_data = data[:,column]
    for i in range(len(column_data)):
        data_i = column_data[i]
        if np.isnan(data_i):
            if label[i] == 1:
                column_data[i] = pos_avg
            elif label[i] == 0:
                column_data[i] = neg_avg
            else:
                print('wrong')
    new_data[:,column] = column_data
#data preprocessing
for column in range(column_num):
    max_value = new_data[:,column].max()
    min_value = new_data[:,column].min()
    new_data[:,column] = (new_data[:,column] - min_value) / (max_value - min_value)

# data split to train and test
data_num = len(new_data)
pos_num = len(disease_data)
neg_num = len(nondisease_data)
pos_weight = neg_num / float(data_num)
neg_weight = pos_num / float(data_num)

random.shuffle(disease_index)
random.shuffle(nondisease_index)
train_pos_index = disease_index[:int(pos_num*0.8)]
test_pos_index = disease_index[int(pos_num*0.8):]
train_neg_index = nondisease_index[:int(neg_num*0.8)]
test_neg_index = nondisease_index [int(neg_num*0.8):]

random.shuffle(test_neg_index)
train_index = np.concatenate((train_neg_index[:len(train_pos_index)], train_pos_index)) #, train_pos_index, train_pos_index,train_pos_index,train_pos_index,train_pos_index))
random.shuffle(train_index)
train_data = new_data[train_index,:]
train_label = label[train_index]
# print train_data.shape
# print len(train_pos_index)
# print len(train_data)
# bb
test_index = np.concatenate((test_neg_index, test_pos_index))
random.shuffle(test_index)
test_data = new_data[test_index,:]
test_label = label[test_index]

#network
Epoch = 1000
Batch_size = 500
Time_step = 1
Input_size = 13
Lr = 1

class txtdataset(Dataset):
    """docstring for txtdataset"""
    def __init__(self, input=None, output=None):
        super(txtdataset, self).__init__()
        self.data = input
        self.label = output

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sequence = self.data[idx,:].astype('float32')
        gt = self.label[idx].astype('float32')
        return sequence, gt

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)

def get_weight(labels):
    labels = labels.numpy()
    totel = 1
    for i in labels.shape:
        totel = i * totel
    pos = np.count_nonzero(labels)
    beta_P = pos_weight
    beta_N = neg_weight #(totel*1.0) / (totel - pos)
    # beta_P =  5.0 * (totel - pos) / totel
    # beta_N =  5.0 * pos / totel

    weight = (1 - labels) * (beta_N) + labels * beta_P
    weight = torch.from_numpy(weight)

    return weight

traindataset = txtdataset(train_data, train_label)
testdataset = txtdataset(test_data, test_label)
train_loader = torch.utils.data.DataLoader(dataset=traindataset, batch_size=Batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=testdataset, batch_size=Batch_size, shuffle=True)

rnn = RNN()
print(rnn)

optimizer = torch.optim.SGD(rnn.parameters(), lr=Lr)   # optimize all parameters
loss_func = nn.BCEWithLogitsLoss()   # the target label is not one-hotted

best_acc = 0.
for epoch in range(Epoch):
    print('train for epoch '+str(epoch))

    if epoch % 100 == 100 - 1:
        lr_ = Lr / 2.
        Lr = Lr / 2.
        print('(poly lr policy) learning rate: ', lr_)
        optimizer = torch.optim.SGD(rnn.parameters(), lr=lr_)

    loss_train = 0.
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 1, Input_size)
        label_step = b_y
        b_y = b_y.view(-1, 1)

        b_x = Variable(b_x)
        b_y = Variable(b_y)

        # print b_x.shape  
        output, weight = rnn(b_x)

        # loss_func = nn.BCEWithLogitsLoss(weight=get_weight(label_step))
 
        # if int(label_step) == 0:
        #     loss = loss_func(output, b_y)
        # elif int(label_step) == 1:
        #     loss = loss_func(output, b_y)
        # y_scores = output.data.view(-1).numpy()
        # y_true = b_y.data.view(-1).numpy()
        # print roc_auc_score(y_true, y_scores)
        # print loss_func(output, b_y)
        loss = loss_func(output, b_y) # + (1 - roc_auc_score(y_true, y_scores)) * 0.9

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_train += loss

    count = 0
    roc = 0.
    error = []
    out_list = []
    label_list = []
    for step, (b_x, b_y) in enumerate(test_loader):
        b_x = b_x.view(-1, 1, Input_size)
        label_step = b_y
        b_y = b_y.view(-1, 1)

        b_x = Variable(b_x)

        output, weight = rnn(b_x)

        for ii in range(output.shape[0]):
            out_list.append(output[ii,:].data.numpy())
            label_list.append(b_y[ii,:].numpy())
            if float(output[ii,:]) >= 0.5 :
                predict = 1
            else:
                predict = 0

            if predict == int(label_step[ii]):
                count += 1
            else:
                error.append(int(label_step[ii]))
        y_scores = output.data.view(-1).numpy()
        y_true = b_y.view(-1).numpy()
        roc += roc_auc_score(y_true, y_scores)

    accuracy = float(count) / (len(test_label))
    roc = roc / (step+1)
    if best_acc < accuracy:
        best_acc =accuracy
        best_epoch = epoch
        best_weight = (weight - weight.min()) / (weight.max() - weight.min())
        best_auc = roc
        # torch.save(rnn,'./best_auc.pth')

    print('loss: '+str(loss_train) + ' acc: ' + str(accuracy))
    print('roc auc: '+str(roc))
    print(error)
print('best epoch: '+str(best_epoch)+' || best_acc: '+str(best_acc)+' || best_auc: '+str(best_auc))
print(best_weight)

### ROC-AUC Curve
fpr, tpr, thresholds = roc_curve(label_list, out_list)

plt.plot(fpr,tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC curve for Heart disease classifier by RNN_LSTM')
plt.xlabel('False positive rate (1-Specificity)')
plt.ylabel('True positive rate (Sensitivity)')
plt.grid(True)

roc_auc_score = sklearn.metrics.roc_auc_score(label_list, out_list)
plt.text(0.95, 0.01, "ROC Curve Covers %i%% Area" % (round(roc_auc_score, 2)*100), verticalalignment='bottom', horizontalalignment = 'right', fontsize = 14)
plt.show()

# plt.savefig('/home/samiulengineer/Desktop/sami/result_image/roc_curve_rnn_lstm.png')


### confusion_matrix

# first convert y_pred_prob to y_prob
for i in range(len(out_list)):
    if out_list[i] >= 0.5:
        out_list[i] = 1.
    else:
        out_list[i] = 0.

# out_list[out_list>=0.5] = 1
# out_list[out_list!=1] = 0

# print (label_list) # y_true/y_test
# print (out_list) # y_predict

cm = confusion_matrix(label_list, out_list)
print('confusion_matrix:', cm)

conf_matrix = pd.DataFrame(data = cm, columns = ['Predicted: 0', 'Predicted: 1'], index = ['Actual: 0', 'Actual: 1'])

fig = sb.heatmap(data = conf_matrix, annot = True, fmt = 'd', cmap = "YlGnBu")
fig.set_title('Confusion Matrix by RNN_LSTM')

# fig.get_figure().savefig('/home/samiulengineer/Desktop/sami/result_image/conf_matrix_rnn_lstm.png')

from sklearn.metrics import classification_report

print(classification_report(label_list, out_list))


### MCC value

from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(label_list, out_list)

print ('MCC : %f' % mcc)


### Specificity Value

TN = cm[0,0]
TP = cm[1,1]
FN = cm[1,0]
FP = cm[0,1]

specificity = TN/(TN+FP)
print ('specificity: %f' % specificity)
