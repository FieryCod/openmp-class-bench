from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import time
import pandas as pd

file_path = "resources/skin1.csv"

start = time.time()
df_raw = pd.read_csv(file_path, delimiter=',')
x_train = []
y_train = []

x_test = []
y_test = []

for i in range(len(df_raw)):
    if i < len(df_raw) * 0.8:
        x_train.append([df_raw['R'][i], df_raw['G'][i], df_raw['B'][i]])
        y_train.append(df_raw['SKIN'][i])
    else:
        x_test.append([df_raw['R'][i], df_raw['G'][i], df_raw['B'][i]])
        y_test.append(df_raw['SKIN'][i])



kn = KNeighborsClassifier(n_neighbors=5, n_jobs=4)
kn.fit(x_train, y_train)
kn_predict = kn.predict(x_test)
kn_acc_score = accuracy_score(y_test, kn_predict)
end = time.time()
print("Total time ", end - start)
print("Acurancy ", kn_acc_score)
