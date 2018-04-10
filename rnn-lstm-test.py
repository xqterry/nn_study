# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import collections
import numpy as np
import matplotlib.pyplot as plt
import argparse
import random


# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('-m', '--mode', type=str, default="gen")
argparser.add_argument('-r', '--restore', action='store_true')
argparser.add_argument('-e', '--epochs', type=int, default=100)
argparser.add_argument('-g', '--genlen', type=int, default=25)
argparser.add_argument('-l', '--layers', type=int, default=1)
argparser.add_argument('-n', '--model', type=str, default='lstm')
args = argparser.parse_args()

# read the fucking data
poetry_file = './data/poetry.txt'

# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8', ) as f:
    for line in f:
        try:
            title, content = line.strip().split(':')
            content = content.replace(' ', '')
            if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
                continue
            if len(content) < 12 or len(content) > 79:
                continue
            content = '[' + content + ']'
            poetrys.append(content)
        except Exception as e:
            pass

# 按诗的字数排序
poetrys = sorted(poetrys, key=lambda line: len(line))
# print('唐诗总数: ', len(poetrys))

# 统计每个字出现次数
all_words = []
for poetry in poetrys:
    all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)

# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [list(map(to_num, poetry)) for poetry in poetrys]
# [[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
# [339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
# ....]

batch_size = 64
n_sent = len(poetrys_vector) // batch_size
data_X = []
data_y = []
for i in range(n_sent):
    si = i * batch_size
    ei = si + batch_size
    bb = poetrys_vector[si:ei]
    maxlen = max(map(len, bb))
    dx = np.full((batch_size, maxlen), word_num_map[' '], np.int32)
    for row in range(len(bb)):
        dx[row, :len(bb[row])] = bb[row]
    dy = np.copy(dx)
    dy[:-1], dy[-1] = dx[1:] , dx[0]
    data_X.append(dx)
    data_y.append(dy)

# print(len(data_X))

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, gpu=False, model="lstm", n_layers=1):
        super(CharRNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gpu = gpu

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

        if gpu:
            self.encoder = self.encoder.cuda()
            self.rnn = self.rnn.cuda()
            self.decoder = self.decoder.cuda()

    def forward(self, input, hidden):
        # print(input, hidden)
        batch_size = input.size(0)
        if hidden == None:
            hidden = self.init_hidden(batch_size)
            
        # print("input shape: ", input.shape)
        
        encoded = self.encoder(input)
        
        # print("embed shape: ", encoded.shape)
        
        encoded = encoded.permute(1, 0, 2)
        # output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output, h0 = self.rnn(encoded, hidden)
        
        le, mb, hd = output.shape
        out = output.view(le * mb, hd)
        
        output = self.decoder(out)
        output = output.view(le, mb, -1)
        output = output.permute(1, 0, 2).contiguous()
        output = output.view(-1, output.shape[2])
        return output, h0

    # def forward2(self, input, hidden):
    #     encoded = self.encoder(input.view(1, -1))
    #     output, hidden = self.rnn(encoded.view(1, 1, -1), hidden)
    #     output = self.decoder(output.view(1, -1))
    #     return output, hidden

    def init_hidden(self, batch_size):
        ret = None
        if self.model == "lstm":
            if self.gpu:
                return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda(),
                 Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)).cuda())

            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                   Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))

        ret = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
        if self.gpu:
            return ret.cuda()
        return ret

def train(input, target):
    hidden = gm.init_hidden(batch_size)
    # hidden = (hidden[0].cuda(), hidden[1].cuda())
    gm.zero_grad()
    loss = 0

    hidden_size = input.shape[1]
    for i in range(hidden_size):
        output, hidden = gm(input[:, i], hidden)
        ov = output.view(batch_size, -1)
        # print('output view size ', ov.size(), " target size 64")
        loss += criterion(ov, target[:, i])

    # output, hidden = gm(input, hidden)
    # loss += criterion(output.view(batch_size, -1), target)

    loss.backward()
    m_opt.step()
    return loss

def train2(input, target, hidden=None):
    
    loss = 0
    
    # print("input size: ", input.shape, " target size: ", target.shape)
    
    output, hidden = gm(input, hidden)
    
    # print("output size: ", output.shape)
    
    loss = criterion(output, target.view(-1))
    
    m_opt.zero_grad()
    loss.backward()
    
    m_opt.step()
    return loss, hidden

def to_word(n):
    for k,v in word_num_map.items():
        if v == n:
            return k
    return ' '

use_gpu = True

input_size = len(words)
hidden_size = 512


gm = None
fn = "%s-test" % args.model
if args.layers > 1:
    fn += "-%d" % args.layers

if args.restore:
    gm = torch.load(fn)
else:
    gm = CharRNN(input_size, hidden_size, input_size, use_gpu, n_layers=args.layers, model=args.model)

# m_opt = torch.optim.Adam(gm.parameters(), lr=0.01)
m_opt = torch.optim.RMSprop(gm.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()#.cuda()

if use_gpu:
    gm = gm.cuda()
    criterion = criterion.cuda()


if args.mode == 'train':
    epochs = args.epochs
    losses = []
    
    for e in range(epochs):
        
        for i in range(len(data_X)):
            x0 = data_X[i]
            x0 = torch.LongTensor(x0)
            y0 = torch.LongTensor(data_y[i])

            x0 = Variable(x0)#.cuda()
            y0 = Variable(y0)#.cuda()

            if use_gpu:
                x0 = x0.cuda()
                y0 = y0.cuda()

            loss, hidden = train2(x0, y0)

            # print(loss.data[0] / 10)
            # break
            losses.append(loss.data[0] / 10)

        print("%d/%d - loss %f" % (e + 1, epochs, np.mean(losses)))
        losses = []

    # plt.plot(losses)
    # plt.show()

    torch.save(gm, fn)

elif args.mode == 'gen':
    print("generate ")
    hidden = gm.init_hidden(1)
    pk = random.choice(list(word_num_map.values()))
    pi = torch.LongTensor(1)
    pi[0] = pk
    pi = pi.unsqueeze(0)
    pi = Variable(pi)

    if use_gpu:
        pi = pi.cuda()

    predicted = "" #to_word(pk)

    output, hidden = gm(pi[:, 0], hidden)
    pi = pi[:, -1]

    for p in range(args.genlen):
        output, hidden = gm(pi, hidden)
        od = output.data.view(-1).div(0.8).exp()
        ti = torch.multinomial(od, 1)[0]

        pi[0] = ti

        predicted += to_word(ti)

    print(predicted)