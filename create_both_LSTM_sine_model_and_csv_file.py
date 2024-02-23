# importing all the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


#__________Generate data for train and test_______


n=500
# Generate 500 data of equal interval b/w 0 to 62.8 
t=np.linspace(0,20.0*np.pi,n)
# create sine wave out of the generated data in radians
X=np.sin(t)

print('print first 5 value of t: \n', t[:5])
print('print first 5 value of X: \n', X[:5])

#set window of past values for LSTM model prediction
window=10




#___________split data into train and test data____________________


last=int(n/5)
Xtrain=X[:-last] # 0 to 400 data as training data

'''
    [:-last] = [0 : -100] = [0:300]
'''

Xtest=X[-last-window:] # 390 to 500 as testing data
'''
    [-last-window :] = [-100-10: ] = [-110: ] = [390: ]
'''

print('\n 1. n: ', n)
print('\n 2. last: ', last)
print('\n 1. length of Xtrain: ', len(Xtrain))
print('\n 2. length of Xtest: ', len(Xtest))
#store window number of points as a sequence

# marking input and output of train data used for training the model
xin=[] # acts as input
next_X=[] # act as ouput for each array of input

print(xin),print(next_X)

# loop starting from 10 to 399
for i in range(window, len(Xtrain)):
    xin.append(Xtrain[i-window:i])
    next_X.append(Xtrain[i])


    '''
    
    when i=10, xin= [ xtrain(10-10:10)] = [xtrain(0:10)] = [ xtrain[0], xtrain[1], xtrain[2],....xtrain[9]  ]
    
    when i=11, xin= [ xtrain(0:10), xtrain(11-10:11)]
                  = [ xtrain(0:10), xtrain(1:11) ]
                  = [ [ xtrain[0], xtrain[1], xtrain[2],....xtrain[9]  ], 
                      [ xtrain[1], xtrain[2], xtrain[3],....xtrain[11] ] ]


    when i=12, xin= [ xtrain(0:10), xtrain(1:11), xtrain(12-10:12)]
                  = [ xtrain(0:10), xtrain(1:11), xtrain(2:12) ]
                  = [ [ xtrain[0], xtrain[1], xtrain[2],....xtrain[9]  ],  
                      [ xtrain[1], xtrain[2], xtrain[3],....xtrain[11] ],
                      [ xtrain[2], xtrain[3], xtrain[4],....xtrain[12] ], ]                      
    
    when i=13, xin= [ xtrain(0:10), xtrain(1:11), xtrain(2:12), xtrain(13-10:13)]
                  = [ xtrain(0:10), xtrain(1:11), xtrain(2:12), xtrain(3:13) ]
                  = [ [ xtrain[0], xtrain[1], xtrain[2],....xtrain[9]  ],  
                      [ xtrain[1], xtrain[2], xtrain[3],....xtrain[11] ],
                      [ xtrain[2], xtrain[3], xtrain[4],....xtrain[12] ],
                      [ xtrain[3], xtrain[4], xtrain[5],....xtrain[13] ] ] 
   
    ........................
    
    when i=399, xin= = [ xtrain(0:10), xtrain(1:11), xtrain(2:12), xtrain(3:13), ....., xtrain(399-10:399) ]
                     = [ xtrain(0:10), xtrain(1:11), xtrain(2:12), xtrain(3:13), ....., xtrain(389:399) ]
                     = [ [ xtrain[0], xtrain[1], xtrain[2],....xtrain[9]  ],  
                         [ xtrain[1], xtrain[2], xtrain[3],....xtrain[11] ],
                         [ xtrain[2], xtrain[3], xtrain[4],....xtrain[12] ],
                         [ xtrain[3], xtrain[4], xtrain[5],....xtrain[13] ],
                         ..........................,
                         [ xtrain[389], xtrain[390], xtrain[391],....xtrain[399] ] 
    '''


    '''
    when i=10, next_X = [ (Xtrain[10]) ]
    when i=11, next_X = [ (Xtrain[10]), (Xtrain[11]) ]
    when i=12, next_X = [ (Xtrain[10]), (Xtrain[11]), (Xtrain[12]) ]
    .....................
    when i=399, next_X = [ (Xtrain[10]), (Xtrain[11]), (Xtrain[12]),........, (Xtrain[399]) ]
    
    '''




    
    
#___________convert the array to format accepted by the LSTM____________________


    # first: conver normal train and test array into numpy array
xin = np.array(xin) 
next_X = np.array(next_X)
print('\n_____________________________Before reshaping_____________________')
print('\n 1. first 5 value of train data is shown below: \n\n', xin[:4])
print('\n 2. dimension of train data is: ', xin.ndim)
print('\n 3. shape of xin before formatting: ', xin.shape )
print('\n 4. what is xin.shape[0]: ', xin.shape[0] )
print('\n 5. what is xin.shape[1]: ', xin.shape[1] )

    # second: Reshape the train data into format for LSTM
xin=xin.reshape(xin.shape[0], xin.shape[1], 1)
print('\n___________________After formatting/reshaping____________________')
print('\n 1. dimension of train data is: ', xin.ndim)
print('\n 2. shape of train data is: ', xin.shape)
print('\n 3. first 5 value of train data is shown below: \n\n', xin[:4])




#____________________Initialize the LSTM model_______________________________
model=Sequential()
model.add(LSTM(units=50,return_sequences=True, input_shape=(xin.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')


#fit LSTM model
history=model.fit(xin, next_X, epochs=50, batch_size=50, verbose=1)

#plot figure
plt.figure()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.semilogy(history.history['loss'])


#generate CSV file___________________________

data={'t':t,'X':X}
df=pd.DataFrame(data)
df.to_csv('LSTM_sine_data.csv',index=False)

model.save('lstm_model_sine_wave_using_csv_file.h5')
    

