# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:37:13 2018

@author: Oppai
"""




from datetime import datetime


def test(model, criterion, testData):
    start = datetime.now()
    
    # load trained model

    #    Set the model to evaluation mode
    model.eval()
    model.cuda()

    correct = 0
    total = 0
    for index, (data, target) in enumerate(testData):
        #        Transfer to gpu
        data, target = data.cuda(), target.cuda()

        out = model(data)
        loss = criterion(out, target)
        predLabel = out.data.max(1, keepdim=True)[1]

        total += data.size()[0]
        correct += predLabel.eq(target.data.view_as(predLabel)).sum()
        print(str(predLabel) + '---> ' + str(target.data.view_as(predLabel)) +
              '(' + str(loss.data) + ')\n')

    print(str(correct) + '\n' + '---------------')

