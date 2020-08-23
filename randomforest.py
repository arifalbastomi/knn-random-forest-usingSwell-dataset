import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time

start_time = time.time()
#load data hapus kolom yang tidak digunakan

train = pd.read_excel('../input/dataswelltrainR.xlsx', sep=';').drop(columns="datasetId").drop(columns="VLF_PCT"). \
    drop(columns="LF_PCT").drop(columns="LF_NU").drop(columns="HF_PCT").drop(columns="HF_NU")
test = pd.read_excel('../input/datatestswellR.xlsx', sep=';').drop(columns="datasetId").drop(columns="VLF_PCT"). \
    drop(columns="LF_PCT").drop(columns="LF_NU").drop(columns="HF_PCT").drop(columns="HF_NU")
target = 'condition'
hrv_features = list(train)
hrv_features = [x for x in hrv_features if x not in [target]]
x_train = train[hrv_features]
y_train = train[target]
x_test = test[hrv_features]
y_test = test[target]

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#
rf = RandomForestClassifier(n_estimators=20, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))
print("Time:", "%s seconds" % (time.time() - start_time))