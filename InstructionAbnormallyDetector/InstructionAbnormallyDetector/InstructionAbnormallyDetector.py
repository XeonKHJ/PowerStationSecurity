import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn

import torch.optim as optim

def fuck(elem):
    return len(elem)


# read data
file = open("..\..\Datasets\data.txt");

dataLists = list()
lineNums = 0;
dataTimestampLengths = list()

isReading = True
singleData = list()
while isReading:
    line = file.readline();
    if line == "":
        isReading = False
    line = line.strip()
    
    if line != "":
        lineDatas = line.split('\t')
        datas = list()
        for i in lineDatas:
            datas.append(int(i))
        singleData.append(datas)
    else:
        dataTimestampLengths.append(len(singleData))
        dataLists.append(singleData)
        singleData = list()

# Pad data

# Sort data first
dataLists.sort(key=fuck, reverse=True)
dataTimestampLengths.sort(reverse=True)
# Padding data
longestSeqLength = len(dataLists[1])
dataBatchSize = len(dataLists)
featureSize = 6

inputTensor = torch.zeros(dataBatchSize,longestSeqLength, featureSize).int()

for i in range(dataBatchSize):
    currentTimeSeq = 1
    for j in range(len(dataLists[i])):
        inputTensor[i][j] = torch.tensor(dataLists[i][j])
        

abbcdd = torchrnn.pack_padded_sequence(inputTensor, dataTimestampLengths, True)

print("")
        