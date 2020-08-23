import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,classification_report
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import sklearn.pipeline
from sklearn.utils import shuffle
import seaborn as sns
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

#nilai k 4 samoai 20 dengan interval 2
k_list = list(range(3,20,3))

#variabel menampung data cv
cv_scores = []

#ANOVA F-value between label/feature for classification tasks.
select = SelectKBest(score_func=f_classif,k=20)

for k in k_list:

    #jarak manhattan p= 1, jarak menggunakan euclidean p=2 ,
    knn = KNeighborsClassifier(n_neighbors=k,p=2)
    #seleksi fitur
    steps = [('feature_selection', select),
             ('model', knn)]
    pipeline = sklearn.pipeline.Pipeline(steps)
    pipeline.fit(x_train, y_train)
    y_predKNN = pipeline.predict(x_test)

    #knn.fit(x_train, y_train)
    #y_predKNN = knn.predict(x_test)

    accuracy = accuracy_score(y_test, y_predKNN)
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_test, y_predKNN))

    print("Time:", "%s seconds" % (time.time() - start_time))
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
print("==========================================")
#membuat grafik crossval
MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15,10))
plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()

best_k = k_list[MSE.index(min(MSE))]
#nilai optimal dari k
print("The optimal number of neighbors is %d." % best_k)








