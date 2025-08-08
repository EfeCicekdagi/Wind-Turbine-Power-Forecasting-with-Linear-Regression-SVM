#SVM model

#Gerekli kütüphanelrin yüklenir
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt 
import pandas as pd
import os

# Veri seti klasörden okunur
folder_path = "archive"
excel_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.csv')]

dfl = []  
for idx, file in enumerate(excel_files, 1): 
    file_path = os.path.join(folder_path, file)
    df_temp = pd.read_csv(file_path)
    df_temp['region'] = idx  
    dfl.append(df_temp)

df = pd.concat(dfl, ignore_index=True)

print(df['region'].value_counts())

#String olan  zaman verisi başka bir veri tipine dönüştürülür
#Zaman verisi daha anlamlı verilere dönüştürülür
df['Time'] = pd.to_datetime(df['Time'])
df['year'] = df['Time'].dt.year
df['month'] = df['Time'].dt.month
df['day'] = df['Time'].dt.day
df['hour'] = df['Time'].dt.hour

X = df.drop(['Power','Time'], axis=1)  
y = df['Power']

#Önce veri seti %80 eğitim %20 test şeklinde bölünür.
#Standartlaştırma işlemleri uygulanır.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()

#SVM modeli eğitilir
svr = SVR(kernel='rbf', C=10.0, epsilon=0.1) 
svr.fit(X_train_scaled, y_train_scaled)
y_pred_scaled = svr.predict(X_test_scaled)

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

print("Test setinde R^2:", r2_score(y_test, y_pred))
print("Test setinde MSE:",mean_squared_error(y_test, y_pred))

#Veri seti tablo şeklinde görüntülenir
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:300], label='Gerçek', color='royalblue')
plt.plot(y_pred[:300], label='SVR Tahmini', color='orange', alpha=0.7)
plt.legend()
plt.xlabel('Veri')
plt.ylabel('Değer')
plt.title('SVR Test Seti (İlk 300 veri)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
