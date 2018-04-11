# -*- coding: utf-8 -*-
# Problem 3
# generate three 2D array for lebel, word number, and word frequency
with open('C:/Onedrive/OneDrive - Knights - University of Central Florida/UCF/Courses/Smart City/HW1/data_task_a/task_a_labeled_train.tf') as df:
    label_list=[]       # define label list   
    word_matrix=[]      # define word number 2D array 
    count_matrix=[]     # define word frequency 2D array  
    for line in df:      # read the raw data line by line   
        word_list=[]     # for each line, define word number list 
        count_list=[]      # for each line, define word frequency list 
        label=line.split(' ')[0]       # pick up the first value as label
        cells=line.split(' ')[1::]     # pick up the the remaining value as list
        for i in range (0, len(cells)):          # read the row data column by column
            word=int(cells[i].split(':')[0])    # for each cell, pick up the left value as word number
            count=int(cells[i].split(':')[1])    # for each cell, pick up the left value as word frequency
            count_list.append(count)             # collect two lists of each line 
            word_list.append(word)          
        label_list.append(label)            # collect label list
        word_matrix.append(word_list)       # collect word number 2D array 
        count_matrix.append(count_list)     # collect word frequency 2D array 
    print (label_list, word_matrix,count_matrix)   


 # convert the data format to be array     
import pandas as pd        
import numpy as np
word_frame=np.array(word_matrix)         
count_frame=np.array(count_matrix) 

# generate the list of word numbers to create a data frame
word_matrix=list(set([j for i in (word_frame) for j in i]))
word_matrix.sort()


# create a data frame of 4000*41675, and fill in all 0
count_output=pd.DataFrame(index=range(0,word_frame.shape[0]), columns=word_matrix)
count_output = count_output.fillna(0)

# fill in the corresponding word frequency based on the comparison between word number and column number
for i in range(0,word_frame.shape[0]):
    for j in word_matrix:   
        for k in range(0,len(word_frame[i])):
            if word_frame[i][k]==j:
                count_output.loc[i][j]=count_frame[i][k]
                j=j+1
print (count_output) 


 # for the test dataset, generate three 2D array for lebel, word number, and word frequency following the same procedure with training dataset
with open('C:/Onedrive/OneDrive - Knights - University of Central Florida/UCF/Courses/Smart City/HW1/data_task_a/task_a_u00_tune.tf') as test:
    label_list_tst=[]
    word_matrix_tst=[]
    count_matrix_tst=[]
    for line in test:
            word_list=[]
            count_list=[]
            label=line.split(' ')[0]       
            cells=line.split(' ')[1::]     
            for i in range (0, len(cells)):          
                word=int(cells[i].split(':')[0])    
                count=int(cells[i].split(':')[1])
                count_list.append(count)
                word_list.append(word)          
            label_list_tst.append(label)
            word_matrix_tst.append(word_list)
            count_matrix_tst.append(count_list)
    print (label_list_tst, word_matrix_tst,count_matrix_tst) 
        
 # convert the data format to be array        
word_frame_tst=np.array(word_matrix_tst)
count_frame_tst=np.array(count_matrix_tst) 

# create a data frame of 4000*41675, and fill in all 0
count_output_tst=pd.DataFrame(index=range(0,word_frame_tst.shape[0]), columns=word_matrix)
count_output_tst = count_output_tst.fillna(0)

# fill in the corresponding word frequency based on the comparison between word number and column number
for i in range(0,word_frame_tst.shape[0]):
    for j in word_matrix:   
        for k in range(0,len(word_frame_tst[i])):
            if word_frame_tst[i][k]==j:
                count_output_tst.loc[i][j]=count_frame_tst[i][k]
                j=j+1
print (count_output_tst) 


# fitting the Naive bayes classifier based on the training dataset
from sklearn.naive_bayes import GaussianNB
x=count_output.as_matrix()
y=np.array(label_list)
nbc=GaussianNB()
nbc.fit(x,y)

# predicting the outcome of test dataset based on the fitted Naive bayes classifier
x_tst=count_output_tst.as_matrix()
pred=nbc.predict(x_tst)

# calculate the accuracy score based on the comparison between test outcome and predicted outcome
from sklearn.metrics import accuracy_score
accuracy_score(np.array(label_list_tst), pred, normalize = True)