import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix

veri = pd.read_csv('data.csv')

print(veri.head())
print(veri.shape)

#<================================ Grafik Çizimleri ==================================>
plt.style.use('ggplot')

sns.pairplot(veri[['chol', 'age', 'trestbps', 'thalach', 'target']], hue='target')
plt.savefig("noktasal_dagilim.png")
plt.show()

plt.figure(figsize=(5,5))

sns.countplot(x="target", data = veri)
plt.ylabel("Sayı")
plt.xlabel("Teşhis")
plt.title('Hasta ve Sağlıklı Birey Dağılımı',fontsize=10)
plt.legend(["Sağlıklı", "Hasta"], title = 'Durum', loc='upper left')
plt.savefig("saglikli_hasta_dagilimi.png")
plt.show()

counts = veri.groupby(['target', 'sex']).target.count().unstack()
counts.plot(kind='bar')
plt.ylabel("Sayı")
plt.xlabel("Cinsiyet")
plt.legend(["Sağlıklı","Hasta"], title = 'Durum', loc='upper right')
plt.title('Cinsiyete Göre Sağlık Dağılımı',fontsize=10)
plt.savefig("cinsiyete_gore_hastalik_dagilimi.png")
plt.show()

sns.distplot(veri["age"], color = "#2258B5", kde=True)
plt.ylabel("Yoğunluk Oranı")
plt.xlabel("Yaş")
plt.legend(["Yaş"], title = 'Değişken', loc='upper left')
plt.title('Genel Yaş Dağılımı', fontsize=10)
plt.savefig("genel_yas_dagilimi.png")
plt.show()

hastalar = veri[veri['target'] == 1]
sns.distplot(hastalar["age"], color = "#2258B5", kde=True)
plt.ylabel("Yoğunluk Oranı")
plt.xlabel("Yaş")
plt.legend(["Yaş"], title = 'Değişken', loc='upper left')
plt.title('Sadece Hasta Olanların Yaş Dağılımı', fontsize=10)
plt.savefig("hasta_yas_dagilimi.png")
plt.show()

sns.countplot(x= hastalar["age"])
plt.ylabel("Kişi Sayısı")
plt.xlabel("Yaş")
plt.legend(["Yaş"], title = 'Değişken', loc='upper left')
plt.title('Sadece Hasta Olanlar için Genişletilmiş Yaş Dağılımı', fontsize=10)
plt.savefig("genisletilmis_hasta_yas_dagilimi.png")
plt.show()



#<==================== Girdi ve Çıktı Değişkenlerinin Tanımlanması ===================>
girdi = np.array(veri.iloc[:,:13].values)
cikti = np.array(veri.iloc[:,-1:].values)


#<================================ Veri Ölçeklendirme ================================>
mm = MinMaxScaler()
girdi = mm.fit_transform(girdi)


#<=========================== Veri Test ve Eğitim Ayrışımı ===========================>
X_train, X_test, y_train, y_test = train_test_split(girdi, cikti, test_size = 0.20, random_state = 0)


#<==================================== Metrikler =====================================>
def degerlendirme_sonuclari(model_adi, karmasiklik_matrisi):
    TP = karmasiklik_matrisi[0][0]
    FP = karmasiklik_matrisi[0][1]
    FN = karmasiklik_matrisi[1][0]
    TN = karmasiklik_matrisi[1][1]
    
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    sensitivity = TP/(TP+FN)
    specifity = TN/(TN+FP)
    accuracy = (TP+TN)/(TP+FP+FN+TN)
    
    sonuc = str(model_adi)+'\n{}\nPrecision:{}\nRecall: {}\nSensitivity: {}\nSpecifity: {}\nAccuracy: {}'.format(29*'-',precision,recall,sensitivity,specifity,accuracy)
    return sonuc


#<================================ Lojistik Regresyon ================================>
logistic = LogisticRegression(max_iter=150)
logistic.fit(X_train, y_train.ravel())

logistic_tahmin = logistic.predict(X_test)

sns.heatmap(confusion_matrix(y_test,logistic_tahmin), annot = True, cmap = "PRGn")
plt.show()

print(degerlendirme_sonuclari("Lojistik Regresyon",confusion_matrix(y_test,logistic_tahmin)))


#<======================================= K-NN =======================================>
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train.ravel())

knn_tahmin = knn.predict(X_test)

sns.heatmap(confusion_matrix(y_test,knn_tahmin), annot = True, cmap = "PRGn")
plt.show()

print(degerlendirme_sonuclari("K-NN",confusion_matrix(y_test,knn_tahmin)))


#<================================ Gaussian Naive Bayes ===============================>
gNB = GaussianNB()
gNB.fit(X_train, y_train.ravel())
    
gNB_tahmin = gNB.predict(X_test)

sns.heatmap(confusion_matrix(y_test,gNB_tahmin), annot = True, cmap = "PRGn")
plt.show()

print(degerlendirme_sonuclari("Gaussian Naive Bayes",confusion_matrix(y_test, gNB_tahmin)))


#<=================================== Karar Ağaçları ==================================>
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X_train, y_train.ravel())
    
decision_tree_tahmin = decision_tree.predict(X_test)

sns.heatmap(confusion_matrix(y_test,decision_tree_tahmin), annot = True, cmap = "PRGn")
plt.show()

print(degerlendirme_sonuclari("Karar Ağaçları",confusion_matrix(y_test, decision_tree_tahmin)))


#<================================= Yapay Sinir Ağları ================================>
ysa = Sequential()
ysa.add(Dense(13, input_dim=13))
ysa.add(Activation("relu"))
ysa.add(Dropout(0.2))
ysa.add(Dense(28))
ysa.add(Activation("relu"))
ysa.add(Dropout(0.2))
ysa.add(Dense(32))
ysa.add(Activation("relu"))
ysa.add(Dense(1))
ysa.add(Activation("sigmoid"))

ysa.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

ysa.fit(X_train, y_train, epochs=100, batch_size=32)
test_loss, test_acc = ysa.evaluate(X_test, y_test)

ysa_tahmin = ysa.predict(X_test)
ysa_binary_tahmin = [1 if i > 0.5 else 0 for i in ysa_tahmin]

sns.heatmap(confusion_matrix(y_test,ysa_binary_tahmin), annot = True, cmap = "PRGn")
plt.show()

print(degerlendirme_sonuclari("Yapay Sinir Ağları",confusion_matrix(y_test, ysa_binary_tahmin)))
