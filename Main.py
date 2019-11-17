import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from scipy.stats import levy
from sklearn.preprocessing import MinMaxScaler
from algorithms import simulated_annealing,GeomDecay, random_hill_climb
from neural import NetworkWeights,ContinuousOpt,gradient_descent
from activation import relu
from sklearn.metrics import accuracy_score
from datetime import datetime
import tkinter as tk
from tkinter import ttk

def get_weights(weights):
    #sws = [np.reshape(weights[0:96],(24,4)),np.reshape(weights[96:100],4),np.reshape(weights[100:116],(4,4)),np.reshape(weights[116:120],4),np.reshape(weights[120:124],(4,1)),weights[124]]
    sws = [np.reshape(weights[0:96],(24,4)),np.reshape(weights[96:100],4),np.reshape(weights[100:116],(4,4)),[0.,0.,0.,0.],np.reshape(weights[116:120],(4,1)),[0.]]
    return sws

def predict(weights,X_test,Y_test,model):
    sws = get_weights(weights)
    model.set_weights(sws)
    y_pred = model.predict(X_test)
    # setting a confidence threshhold of 0.9
    y_pred_labels = list(y_pred > 0.9)

    for i in range(len(y_pred_labels)):
        if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
        else : y_pred_labels[i] = 0
    return accuracy_score(Y_test,y_pred_labels)

def BackPropagation():
    model = Sequential()
    model.add(Dense(4,input_dim=X_train.shape[1]))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model1 = model
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(X_train,Y_train,batch_size=2,epochs=50,validation_data = (X_test,Y_test))
    y_pred = model.predict(X_test)
    # setting a confidence threshhold of 0.9
    y_pred_labels = list(y_pred > 0.9)

    for i in range(len(y_pred_labels)):
        if int(y_pred_labels[i]) == 1 : y_pred_labels[i] = 1
        else : y_pred_labels[i] = 0
    return accuracy_score(Y_test,y_pred_labels)


def EagleStrategyWithSA(model,iters,rate):
    mn = MinMaxScaler(feature_range=(-3,3))
    r = levy.rvs(size=iters)
    max_accuracy = 0
    final_weights = []
    count = 1
    for i in list(r):
        weights = np.random.uniform(-1,1,120)
        weights = weights * i
        weights = mn.fit_transform(weights.reshape(-1,1))
        acc0 = predict(weights,X_test,Y_test,model)
        if acc0 > 0.5:

            clip_max = 3
            fitness = NetworkWeights(X_train,Y_train,[25,4,4,1],relu,True,True,rate)
            problem = ContinuousOpt(120,fitness,maximize=False,min_val=-1*clip_max,max_val=clip_max,step=rate)
            #print(problem.get_length(),len(weights))
            fitted_weights, loss = simulated_annealing(problem,schedule=GeomDecay(),max_attempts=50,max_iters=500,init_state=weights,curve=False)
            acc = predict(fitted_weights,X_test,Y_test,model)
            if acc > max_accuracy:
                #print(acc0)
                max_accuracy = acc
                final_weights = fitted_weights
            if max_accuracy > 0.99:
                break
            count+=1
    return max_accuracy,loss,count,final_weights  

def EagleStrategyWithHill(model,iters,rate):
    mn = MinMaxScaler(feature_range=(-3,3))
    r = levy.rvs(size=iters)
    max_accuracy = 0
    final_weights = []
    count = 1
    for i in list(r):
        weights = np.random.uniform(-1,1,120)
        weights = weights * i
        weights = mn.fit_transform(weights.reshape(-1,1))
        acc = predict(weights,X_test,Y_test,model)
        if acc > 0.5:
            clip_max = 5
            fitness = NetworkWeights(X_train,Y_train,[25,4,4,1],relu,True,True,rate)
            problem = ContinuousOpt(120,fitness,maximize=False,min_val=-1*clip_max,max_val=clip_max,step=rate)
            #print(problem.get_length(),len(weights))
            fitted_weights, loss = random_hill_climb(problem, max_attempts=10, max_iters=1000, restarts=0,
                      init_state=weights, curve=False, random_state=None)
            acc = predict(fitted_weights,X_test,Y_test,model)
            if acc > max_accuracy:
                max_accuracy = acc
                final_weights = fitted_weights
            if max_accuracy > 0.99:
                break
            count+=1
    return max_accuracy
#Reading the file and setting the samples
df = pd.read_csv('Eyes.csv')
X = df[df.columns[1:25]]
Y = df[df.columns[25]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.10,random_state=42)
#Normalizing the data
sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

#Defining Model
model = Sequential()
model.add(Dense(4,input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#Tkinter functions

def SA():
    start = datetime.now()
    accuracy,count,final_weights = EagleStrategyWithSA(model,40,0.2)
    end=datetime.now()-start
    master = tk.Tk()
    master.title("Simulated Annealing Results")
    #x = sensor_value #assigned to variable x like you showed
    master.minsize(width=400,height=50)
    w = tk.Label(master, text="Time taken : "+str(end)+"\n"+"Accuracy : "+str(accuracy)) #shows as text in the window
    #w = tk.Label(master, text="Accuracy : "+str(accuracy))
    w.pack() #organizes widgets in blocks before placing them in the parent.          
    master.mainloop()

    
def rhc():
    start = datetime.now()
    accuracy,count,final_weights = EagleStrategyWithHillClimbing(model,40,0.2)
    end=datetime.now()-start
    master = tk.Tk()
    master.title("Random Hill Climbing Results")
    #x = sensor_value #assigned to variable x like you showed
    master.minsize(width=400, height=50)
    w = tk.Label(master, text="Time taken : "+str(end)+"\n"+"Accuracy : "+str(accuracy)) #shows as text in the window
    #w = tk.Label(master, text="Accuracy : "+str(accuracy))
    w.pack() #organizes widgets in blocks before placing them in the parent.          
    master.mainloop()
    
def backprop():
    startTime=datetime.now()
    accuracy = BackPropagation()
    end=datetime.now()-startTime
    master = tk.Tk()
    master.title("Base Sklearn Results")
    master.minsize(width=400,height=50)
    w = tk.Label(master, text="Time taken : "+str(end)+"\n"+"Accuracy : "+str(accuracy)) #shows as text in the window
    w.pack() #organizes widgets in blocks before placing them in the parent.          
    master.mainloop()
    
r = tk.Tk() 
r.title('Choose Algorithm') 
button1 = tk.Button(r, text='Simulated Annealing', width=25, command=simulated) 
button2 = tk.Button(r, text='Random Hill Climb', width=25, command=rhc)
button3 = tk.Button(r, text='BackPropagation', width=25, command=backprop)
button1.pack() 
button2.pack()
button3.pack()
r.mainloop() 
