import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('diabetes_data.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.shape)
columns=data.columns.values
#for i in columns:
    #sn.boxplot(data[i])
    #plt.show()
    #if len(data[i].value_counts().values) <=5:
    #    sn.countplot(data[i])
    #    plt.show()
#mentall health
#physical health
data['z-scores']=(data.PhysHlth-data.PhysHlth.mean())/data.PhysHlth.std()
df=data[(data['z-scores'] > -3)&(data['z-scores'] <3)]
qa1=df.PhysHlth.quantile(0.25)
qa3=df.PhysHlth.quantile(0.75)
iqr=qa3-qa1
up=qa3+1.5*iqr
lo=qa1-1.5*iqr
df=df[(df.PhysHlth <up)&(df.PhysHlth >lo)]
qua1=df.PhysHlth.quantile(0.25)
qua3=df.PhysHlth.quantile(0.75)
iQr=qua3-qua1
upp=qua3+1.5*iQr
low=qua1-1.5*iQr
df=df[(df.PhysHlth <upp)&(df.PhysHlth >low)]
df['z-scores']=(df.BMI-data.BMI.mean())/data.BMI.std()
df=df[(df['z-scores'] > -3)&(df['z-scores'] <3)]
print(df.shape)
print(data.shape)
q1=df.BMI.quantile(0.25)
q3=df.BMI.quantile(0.75)
iQR=q3-q1
upper=q3+1.5*iQR
lower=q1-1.5*iQR
df=df[(df.BMI < upper) &(df.BMI >lower)]
qa1=df.BMI.quantile(0.25)
qa3=df.BMI.quantile(0.75)
Iqr=qa3-qa1
u=qa3+1.5*Iqr
l=qa1-1.5*Iqr
df=df[(df.BMI < u)&(df.BMI >l )]
x=df[['Age', 'Sex', 'HighChol', 'CholCheck', 'BMI', 'Smoker',
       'HeartDiseaseorAttack', 'PhysActivity', 'Fruits', 'Veggies',
       'HvyAlcoholConsump', 'GenHlth', 'PhysHlth', 'DiffWalk',
       'Stroke', 'HighBP']]
y=df['Diabetes']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y)
from sklearn.linear_model import LogisticRegression
lg=LogisticRegression()
lg.fit(x_train,y_train)
log_predict=lg.predict(x_test)
from sklearn.tree import DecisionTreeClassifier
dtree_classi=DecisionTreeClassifier()
dtree_classi.fit(x_train,y_train)
dtree_predict=dtree_classi.predict(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn_classi=KNeighborsClassifier()
knn_classi.fit(x_train,y_train)
knn_pred=knn_classi.predict(x_test)
print('The Logistic regression score-> ',lg.score(x_test,y_test))
print('The Decision Tree score-> ',dtree_classi.score(x_test,y_test))
print('The KNN classification score is-> ',knn_classi.score(x_test,y_test))
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(log_predict,marker='o',color='blue',label='logistic_reg_prediction')
plt.title('Logistic regression prediction vs Y_test')
plt.legend()
plt.show()
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(knn_pred,marker='o',color='blue',label='KNN_Classification_prediction')
plt.title('KNN Classification prediction vs Y_test')
plt.legend()
plt.show()
plt.plot(y_test,marker='o',color='red',label='y_test')
plt.plot(dtree_predict,marker='o',color='blue',label='decision_tree_prediction')
plt.title('Decision tree classification prediction vs Y_test')
plt.legend()
plt.show()
from keras.models import Sequential
from keras.layers import Dense
import keras.activations,keras.losses
models=Sequential()
models.add(Dense(units=x.shape[1],input_dim=x_train.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=x.shape[1],activation=keras.activations.sigmoid))
models.add(Dense(units=1,activation=keras.activations.sigmoid))
models.compile(optimizer='adam',metrics='accuracy',loss=keras.losses.binary_crossentropy)
hist=models.fit(x_train,y_train,batch_size=20,epochs=200,validation_split=0.45)
plt.plot(hist.history['accuracy'],label='training accuracy',marker='o',color='red')
plt.plot(hist.history['val_accuracy'],label='val_accuracy',marker='o',color='blue')
plt.title('Training Vs  Validation accuracy')
plt.legend()
plt.show()