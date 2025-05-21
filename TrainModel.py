from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Veriyi eğitim ve test olarak ayır (80% eğitim, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli tanımla ve eğit
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Tahmin yap
y_pred = model.predict(X_test)

# Sonuçları yazdır
print("Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("\nSınıflandırma Raporu:\n", classification_report(y_test, y_pred, target_names=["genuine", "spoof"]))
