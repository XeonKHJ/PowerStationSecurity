import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as torchrnn

import torch.optim as optim
from LSTMAbnormallyDetector import LstmRNN

def fuck(elem):
    return len(elem)


# read data
file = open("..\..\Datasets\data.txt");

dataLists = list()
lineNums = 0;

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
        dataLists.append(singleData)
        singleData = list()

paddedTrainingSet, dataTimestampLengths = LstmRNN.PadData(dataLists, 6)

# Padding data
longestSeqLength = len(dataLists[1])
dataBatchSize = len(dataLists)


lstm_model = LstmRNN(6, 6, 6, 2)
#lstm_model.forward(inputTensor.float())

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-2)

max_epochs = 10000
for epoch in range(max_epochs):
    output = lstm_model(paddedTrainingSet, dataTimestampLengths)
    loss = loss_function(output, paddedTrainingSet)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if loss.item() < 1e-4:
        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, max_epochs, loss.item()))
        print("The loss value is reached")
        break
    elif (epoch+1) % 100 == 0:
        print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch+1, max_epochs, loss.item()))


trainingExample = paddedTrainingSet[0][:]
# prediction on training dataset
predictive_y_for_training = lstm_model(paddedTrainingSet, dataTimestampLengths)
predictive_y_for_training = predictive_y_for_training.view(-1, OUTPUT_FEATURES_NUM).data.numpy()



print("")
        