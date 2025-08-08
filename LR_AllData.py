#Linear Regression

#Gerekli kütüphanelrin yüklenir
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


folder_path = "archive"
excel_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

dfl = []  
for idx, file in enumerate(excel_files, 1): 
    file_path = os.path.join(folder_path, file)
    df_temp = pd.read_csv(file_path)
    df_temp['region'] = idx 
    dfl.append(df_temp)

df = pd.concat(dfl, ignore_index=True)

#String olan  zaman verisi başka bir veri tipine dönüştürülür
#Zaman verisi daha anlamlı verilere dönüştürülür
df['Time'] = pd.to_datetime(df['Time'])
df['year'] = df['Time'].dt.year
df['month'] = df['Time'].dt.month
df['day'] = df['Time'].dt.day
df['hour'] = df['Time'].dt.hour

x = df.drop(['Power','Time'], axis=1)  
y = df['Power']

#Linear Regresyon modeli işlemleri :
#Önce veri seti %80 eğitim %20 test şeklinde bölünür.
#Model eğitilir.
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
#Veri seti standartlaştırılıp yeni bir model üzerinden tekrar değerlendirilmiştir.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scaled,y_train)

y_pred2 = model2.predict(x_test_scaled)

mse = mean_squared_error(y_test, y_pred2)
r2 = r2_score(y_test, y_pred2)

print("Mean Squared Error (MSE):", mse)
print("R^2 Score:", r2)

#Veri seti tablo şeklinde görüntülenir
plt.figure(figsize=(10,4))
plt.plot(y_test.values[:300],label='Gerçek ',color='royalblue')
plt.plot(y_pred2[:300], label='Linear Regresyon Tahmini',color='orange', alpha=0.7)
plt.legend()
plt.xlabel('Veri')
plt.ylabel('Değer')
plt.title('Linear Regresyon Test Seti (İlk 300 veri)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
