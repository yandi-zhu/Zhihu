
# coding: utf-8

import pickle
import pandas as pd
import torch
from torch import nn,optim
import torch.nn.functional as F
import numpy as np
import sys
import gc
import time
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.metrics import roc_auc_score
print('Start!',flush=True)
def sizemb(data):
    return sys.getsizeof(data)/1024**2

def parse_str(d):
    return list(map(float, d.split()))
def parse_list_1(d):
    if d == '-1':
        return ['0']
    return list(map(lambda x: int(x[1:]), str(d).split(',')))
def parse_list_2(d):
    if d == '-1':
        return [0]
    return list(map(lambda x: int(x[2:]), str(d).split(',')))
def parse_map(d):
    if d == '-1':
        return {}
    else:
        pairs = []
        for z in d.split(','):
            p = z.split(':')
            pairs.append([p[0],float(p[1])])
        return(dict(pairs))

word_dict = {}
with open("../word_vectors_64d.txt") as f:
    for line in f.readlines():
        (word,embed)=line.split('\t')
        word_dict[word] = parse_str(embed)

topic_dict = {}
with open('../topic_vectors_64d.txt') as f:
    for line in f.readlines():
        (topic,embed)=line.split('\t')
        topic_dict[topic] = parse_str(embed)
        
#有些问题没有标题、描述、主题；
Q_info = pd.read_csv('../question_info_0926.txt',sep='\t',names=['question_id', 'question_time', 'title_sw_series', 'title_w_series', 'desc_sw_series', 'desc_w_series', 'topic'])

Q_info.set_index('question_id',drop=True,inplace=True)

Q_A_pair = pd.read_csv('../invite_info_0926.txt',sep='\t',names=['question_id', 'author_id', 'invite_time', 'label'])

Q_A_pair_eval = pd.read_csv('../invite_info_evaluate_1_0926.txt',sep='\t',names=['question_id', 'author_id', 'invite_time', 'label'])

A_info = pd.read_csv('../member_info_0926.txt',sep='\t',names=['author_id', 'gender', 'keyword', 'grade', 'hotness', 'reg_type','reg_plat','freq',
                                 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2', 'E2',
                                 'score', 'topic_attent', 'topic_interest'])

A_info.drop(columns=['keyword', 'grade', 'hotness', 'reg_type','reg_plat'],inplace=True)

A_info.topic_interest = A_info.topic_interest.apply(parse_map)

with open('./author_count.pkl', 'rb') as file:
    author_count = pickle.load(file)

A_info = A_info.merge(author_count,how='left',left_on='author_id',right_index=True)

A_info.fillna(0,inplace=True)

A_info.set_index('author_id',inplace=True,drop='True')

del author_count
gc.collect()

def wid_value(wids):
    #根据词的id list 返回 对应词embedding list
    #没有返回array[None]
    if wids == '-1':
        return [[0]*64]
    else:
        return [word_dict.get(i) for i in wids.split(',')]
    
#主题是没有时序性质的 可以直接放进去 或者进行点积
def tid_value(tids):
    if tids == '-1':
        return [[0]*64]
    else:
        return np.array([topic_dict.get(i) for i in tids.split(',')])

device = torch.device('cpu')
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#标题信息提取。不等长seq2one
def get_Q_info(Q_A_pair,info,device):
    #获取提问对应的标题信息；返回pack_padded的Tensor及原始顺序
    n = len(Q_A_pair)
    if info in ['title_w_series','desc_w_series']:
        w_series = Q_info.loc[Q_A_pair.question_id,info].apply(wid_value).values.tolist()
    elif info =='topic':
        w_series = Q_info.loc[Q_A_pair.question_id,info].apply(tid_value).values.tolist()
    w_series = [torch.tensor(i,dtype=torch.float,device=device) for i in w_series]
    w_series_idx = list(zip(range(n),[i.shape[0] for i in w_series]))
    w_series_idx.sort(key=lambda x:x[1],reverse=True)
    w_series.sort(key=lambda x:x.shape[0],reverse=True)
    pped = pack_padded_sequence(pad_sequence(w_series,batch_first=True),[i[1] for i in w_series_idx],batch_first=True)
    return pped,torch.tensor([i[0] for i in w_series_idx],device=device)


def pad_output(pped,idx):
    #pad output，并按原始顺序返回
    ped,length = pad_packed_sequence(pped,batch_first=True)
    outp = []
    for i in range(len(length)):
        outp.append(ped[i,length[i]-1,:])
    return torch.stack(outp).index_select(0,idx)


A_object_feature = ['gender', 'freq', 'A1', 'B1', 'C1', 'D1', 'E1', 'A2', 'B2', 'C2', 'D2','E2']
#去掉一些定量变量 
A_num_feature = ['score','excellent','recommend', 'round_table', 'figure', 'video',
                 'num_word', 'num_like','num_unlike', 'num_comment', 'num_favor', 'num_thank', 'num_report',
                'num_nohelp', 'num_oppose', 'answer_num']
A_topic_feature = ['topic_attent', 'topic_interest']


#用户定量变量 维数
A_object_dims = A_info[A_object_feature].apply(lambda x: len(x.unique()),axis = 0).values
#数字标注
A_object_info = A_info[A_object_feature].apply(LabelEncoder().fit_transform)
#累加
A_object_info = A_object_info.add(np.array((0,*np.cumsum(A_object_dims)[:-1]),dtype=np.long))


A_num_info = A_info[A_num_feature].copy()

#Q_id & A_id
LE_Q = LabelEncoder()
Q_LE = LE_Q.fit_transform(pd.concat([Q_A_pair.question_id,Q_A_pair_eval.question_id],axis=0))

Q_A_pair['Q_LE'] = Q_LE[:len(Q_A_pair)]
Q_A_pair_eval['Q_LE'] = Q_LE[len(Q_A_pair):]

LE_A = LabelEncoder()
A_LE = LE_A.fit_transform(pd.concat([Q_A_pair.author_id,Q_A_pair_eval.author_id],axis=0))

Q_A_pair['A_LE'] = A_LE[:len(Q_A_pair)]
Q_A_pair_eval['A_LE'] = A_LE[len(Q_A_pair):]

Q_id_num = len(LE_Q.classes_)
A_id_num = len(LE_A.classes_)

def get_A_info(Q_A_pair,device):
    #input batch Q_A_pair
    #return batch A_num/A_object
    A_idx = Q_A_pair.author_id.values
    A_num = torch.tensor(A_num_info.loc[A_idx].values,dtype=torch.float,device=device)
    A_object = torch.tensor(A_object_info.loc[A_idx].values,dtype=torch.long,device=device)
    return A_num,A_object


class Inception(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(1,64),stride=1)
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(2,64),stride=1)
        self.conv3 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,64),stride=1)
    def forward(self,x):
        n = x.shape[0]
        x1 = self.conv1(x).squeeze(3)
        x2 = self.conv2(x).squeeze(3)
        x3 = self.conv3(x).squeeze(3)
        x = torch.cat((x1,x2,x3),dim=2).view(n,-1)
        return x #(batch_size*48)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix
    
class mynet(nn.Module):
    def __init__(self,A_object_dims,Q_id_num,A_id_num,device):
        super(mynet,self).__init__()
        self.rnn_title_w = nn.LSTM(64,64,3,batch_first=True)
        self.rnn_title_t = nn.LSTM(64,64,1,batch_first=True)
        self.em_A = nn.Embedding(A_object_dims,8)
        self.em_A_dense = nn.Sequential(nn.Linear(12*8,64),
                                       nn.PReLU(),
                                       nn.Linear(64,48),
                                       nn.PReLU())
        self.A_num_BN = nn.BatchNorm1d(16)
        self.conv = Inception()
        
        self.em_Qid = nn.Embedding(Q_id_num,8)
        self.em_Aid = nn.Embedding(A_id_num,8)
        self.FM = FactorizationMachine()
        self.out = nn.Sequential(nn.Linear(48,32),
                                 nn.PReLU(),
                                 nn.Linear(32,16),
                                 nn.PReLU(),
                                 nn.Linear(16,1))
        torch.nn.init.xavier_uniform_(self.em_A.weight.data)
        torch.nn.init.xavier_uniform_(self.em_Aid.weight.data)
        torch.nn.init.xavier_uniform_(self.em_Qid.weight.data)
        self.device = device
    def forward(self,Q_A_pair):
        batch_size = len(Q_A_pair)
        #Q title word
        title_w,title_w_idx = get_Q_info(Q_A_pair,'title_w_series',self.device)
        title_w_out,_ = self.rnn_title_w(title_w)
        title_w_out = pad_output(title_w_out,title_w_idx) #(batch*LSTM(hid))
        #Q topic
        title_t,title_t_idx = get_Q_info(Q_A_pair,'topic',self.device)
        title_t_out,_ = self.rnn_title_t(title_t)
        title_t_out = pad_output(title_t_out,title_t_idx) #(batch*LSTM(hid))
        #A num(batch*11); 
        A_num,A_object = get_A_info(Q_A_pair,self.device)
        A_num = self.A_num_BN(A_num)
        A_object2 = self.em_A(A_object) #FM用
        A_object = A_object2.reshape(batch_size,-1) #(batch*(12*em_A))
        A_object = self.em_A_dense(A_object)
        A_feature = torch.cat((A_num,A_object),dim=1)
        #Q_topic,Q_title,A_info
        combined1 = torch.stack((title_w_out,A_feature,title_t_out),dim=1).unsqueeze(1) #(batch*1*3*64)
        combined1 = self.conv(combined1)
        out1 = self.out(combined1)
        
        #Q_id & A_id
        Q_id = self.em_Qid(torch.tensor(Q_A_pair.Q_LE.values,dtype=torch.long,device=self.device)).unsqueeze(1) #(batch*1*8)
        A_id = self.em_Aid(torch.tensor(Q_A_pair.A_LE.values,dtype=torch.long,device=self.device)).unsqueeze(1) #(batch*1*8)
        combined2 = torch.cat((Q_id,A_id,A_object2),dim=1)
        out2 = self.FM(combined2)
        #Total
        
        return out1+out2/2

def AUC(y,y_hat):
    '''
    y_hat: net output
    y: true label, series
    '''
    with torch.no_grad():
        y_hat = torch.sigmoid(y_hat).cpu().numpy().squeeze()
        y_true = y.cpu().numpy()
    return roc_auc_score(y_true,y_hat)

net = mynet(int(sum(A_object_dims)),Q_id_num,A_id_num,device)
net.to(device)


batch_size = 1024
lr = 0.01
epoch = 10
loss = nn.BCEWithLogitsLoss()


from sklearn.model_selection import ShuffleSplit

SSplit = ShuffleSplit(n_splits=1,train_size=0.8,test_size=0.2)

#划分训练集和测试集
for train_idx,test_idx in SSplit.split(Q_A_pair):
    train_X = Q_A_pair.loc[train_idx]
    train_y = Q_A_pair.loc[train_idx,'label']
    test_X = Q_A_pair.loc[test_idx]
    test_y = Q_A_pair.loc[test_idx,'label']


def generate_data(train_X,train_y,batch_size):
    n = train_X.shape[0]
    totali = n//batch_size
    i = 0
    while i <= totali:
        if i == totali:
            idx = slice(i*batch_size,n)
        else:
            idx = slice(i*batch_size,(i+1)*batch_size)
        data_X = train_X[idx]
        data_y = torch.tensor(train_y[idx].values,dtype=torch.float,device=device)
        i = i + 1
        yield data_X,data_y

def train(train_X,train_y,test_X,test_y,net,loss,batch_size,epoch,lr):
    train_l,test_l = [],[]
    optimizer = optim.Adam(net.parameters(),lr,weight_decay=0)
    for i in range(epoch):
        time_1 = time.time()
        train_li = []
        j = 0
        net.train()
        for X,y in generate_data(train_X,train_y,batch_size):
            optimizer.zero_grad()
            y_hat = net(X)
            l = loss(y_hat.view(-1),y)  
            l.backward()
            train_li.append(l.cpu().item())
            optimizer.step()
            if(j%100==0):
                print('Check:%d, Train Loss:%f'%(j//100,np.mean(train_li)),flush=True)
                print('AUC:%f'% AUC(y,y_hat))
            j += 1
        print('Epoch:%d, Train Loss:%f'%(i,np.mean(train_li)),flush=True)
        train_l.append(np.mean(train_li))
        torch.save(net.state_dict(),'%s/%snet.pt'%(PATH,str(i)))
        if test_y is not None:
            test_li = []
            auc = []
            net.eval()
            for X,y in generate_data(test_X,test_y,batch_size):
                y_hat = net(X)
                li = loss(y_hat.view(-1),y).cpu().item()
                auci = AUC(y,y_hat)
                test_li.append(li)
                auc.append(auci)
            print('Epoch:%d, Valid Loss:%f, AUC:%f '%(i,np.mean(test_li),np.mean(auc)),flush=True)
        print('Epoch:%d, Time used:%.4f'%(i,time.time()-time_1),flush=True)
        test_l.append(np.mean(test_li))
    return train_l,test_l

PATH = '../model2'
train_l,test_l = train(train_X,train_y,test_X,test_y,net,loss,batch_size,epoch,lr)

