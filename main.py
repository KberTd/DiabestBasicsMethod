import pandas as pd #pandas veri işlemesi ve analiz için ihtiyaç kullanılan yazılım kütüphanesidir.
from sklearn.model_selection import train_test_split #sklearn machine learning için kullanılıyor
from sklearn.impute import SimpleImputer #datayı satır ve sutün olarak dökmek için kullanılıyor
from sklearn import metrics #metrics sklearn kütüphanesinde bir classtır. accuracy_score fonksiyonu Classifierin accuracysini hesaplar
from sklearn.ensemble import RandomForestClassifier #hızlı basit ve esnek olduğu için bunu kullanıyoruz


df = pd.read_csv("C:\\Users\\ARDA\\PycharmProjects\\pythonProjectLastDance\\main.py")


X = df.drop('Outcome', axis=1) #değişkenlerin drop üzerinde görüntülenmesi.  axis  x 1 axis y 0
y = df['Outcome'] #OUTCOME ÇIKTISI OLARAK AYIRMA X VE Y AXİS= 1 SÜTUNDUR AXİS= 0 SATIRDIR

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.33,
                                                    random_state=7)
#veri seti öncelikle train ve test olarak ikiye ayrılıyor.
#x = bağımsız değişken y = hedef(target) değişken test size = 0.33 test için ayrılan küme yüzdesi

fill_values = SimpleImputer(missing_values=0, strategy='mean') #eksik değerler için kullandık
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
#satır sonucu elde etmek için her modelden yapılan tahminler birleştirilir.
# ( normalde 2 değerden sadece 1 tanesi fit lenir 2 değerin fitlenmesi problemlere yol açabilir ama bizim uygulamamızda probleme yol açmadı.)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(df)
print("Random Forest Accuracy_Score =", format(metrics.accuracy_score(y_test, predictions)))
#rfc.fit  x train ve y traini rfc ye uyumlandırıyor.
# Eğer etiketleri değil de olasılıkları öğrenmek istiyorsak predict_proba fonksiyonunu kullanmamız gerekiyor.
# Sonuçlar her etikete ait olma yüzdesini içeriyor.
