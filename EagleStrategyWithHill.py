import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SC
import numpy as np

df = pd.read_csv('Eyes.csv')
X = df[df.columns[1:25]]
Y = df[df.columns[25]]
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.10,random_state=42)

sc = SC()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
X_train, X_test, Y_train, Y_test = np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)

"""from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation

model = Sequential()
model.add(Dense(4,input_dim=X_train.shape[1]))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('relu'))
model.add(Dense(4))
model.add(Activation('sigmoid'))
opt = 'adam'
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
"""
from scipy.stats import levy
import mlrose


r = levy.rvs(size=1)
weights = np.random.uniform(-1,1,116)
weights = weights * r
weights = sc.fit_transform(weights.reshape(-1,1))

print(model.set_weights(weights))





