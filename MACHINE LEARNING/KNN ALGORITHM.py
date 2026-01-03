import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:\Users\Akshaishri\Downloads\weather-weka.csv')
df.columns=df.columns.str.strip()
df=df.dropna(subset=['outlook'])
X=df[['temperature','humidity']].values
y_raw=df['outlook'].values
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
a=int(input("Enter the temperature"))
b=int(input("Enter the humidity"))
new_day=np.array([[a,b]])
predicted_label=knn.predict(new_day)
predicted_weather=label_encoder.inverse_transform(predicted_label)
colors=['gray','blue','orange']
labels=label_encoder.classes_
plt.figure(figsize=(8,6))
for i,label in enumerate(labels):
    plt.scatter(X[y==i,0],X[y==i,1],c=[colors[i]],label=label,edgecolor='k',s=100)
plt.scatter(new_day[0,0],new_day[0,1],c='red',marker='*',s=200,
            label=f'NewDay(Predicted.{predicted_weather[0]}')
plt.xlabel('Temperature(c)')
plt.ylabel('Humidity(%)')
plt.title('Weather condition prediction with KNN')
plt.legend()
plt.show()
