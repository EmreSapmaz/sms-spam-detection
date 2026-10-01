import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os  # Dosya kontrolü için
import joblib  # Model kaydetme için
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk

# Sklearn Kütüphaneleri
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLTK indirmeleri
nltk.download('stopwords', quiet=True)

def clean_text(text):
    """
    Hafta 3: Veri Temizleme Pipeline'ı
    """
    ps = PorterStemmer()
    text = str(text).lower() 
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    all_stopwords = stopwords.words('english')
    words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(words)

def run_training_pipeline():
    # 1. Verileri Yükle ve Birleştir
    print("--- Veriler Hazırlanıyor ---")
    
    # A) Ham veriyi (Normal Mesajlar) al
    df_raw = pd.read_csv('data/spam_cleaned.csv')
    df_ham = df_raw[df_raw['label'] == 'ham'].copy()
    df_ham['target'] = 'Normal' 
    
    # B) Kümelenmiş Spam verisini al 
    df_clustered_all = pd.read_csv('data/spam_clustered.csv')
    df_spam = df_clustered_all[df_clustered_all['label'] == 'spam'].copy()
    
    # Etiketleme
    label_map = {
        0: 'Abonelik_Hizmet',
        1: 'Dolandiricilik_Odul',
        2: 'Bahis_Yarisma',
        3: 'Cihaz_Promosyon'
    }
    df_spam['target'] = df_spam['cluster'].map(label_map)
    
    # Ana Veri Setini Oluştur
    df_final = pd.concat([df_ham[['message', 'target']], df_spam[['message', 'target']]])
    
    # --- GERİ BİLDİRİM OKUMA (ACTIVE LEARNING) ---
    feedback_path = 'data/feedback_log.csv'
    if os.path.exists(feedback_path):
        try:
            print(f"\n📢 Geri bildirim dosyası bulundu! Model yeni verilerle güçlendiriliyor...")
            df_feedback = pd.read_csv(feedback_path)
            
            # Sütun isimlerini eşle
            df_feedback.rename(columns={'Message': 'message', 'True_Label': 'target'}, inplace=True)
            df_feedback = df_feedback[['message', 'target']]
            
            print(f"   Eklenen yeni veri sayısı: {len(df_feedback)}")
            df_final = pd.concat([df_final, df_feedback], ignore_index=True)
            
        except Exception as e:
            print(f"⚠️ Geri bildirim dosyası okunurken hata oluştu: {e}")
    else:
        print("\nℹ️ Henüz geri bildirim verisi yok. Standart eğitim yapılıyor.")
    # -------------------------------------------------------------------

    # NaN kontrolü
    df_final.dropna(subset=['target'], inplace=True)
    
    print(f"\nEğitim için Toplam Veri: {len(df_final)}")
    
    # 2. Metin Temizliği
    print("Metinler temizleniyor...")
    df_final['clean_text'] = df_final['message'].apply(clean_text)
    
    # 3. Vektörleştirme
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df_final['clean_text']).toarray()
    y = df_final['target']
    
    # 4. Eğitim/Test Ayırma
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # 5. Modellerin Tanımlanması
    custom_weights = {
        'Normal': 1,
        'Abonelik_Hizmet': 15, 'Dolandiricilik_Odul': 20, 
        'Bahis_Yarisma': 15, 'Cihaz_Promosyon': 15
    }
    
    # Random Forest'ı ayrıca değişkende tutuyoruz ki kesin onu kaydedelim
    rf_model = RandomForestClassifier(class_weight=custom_weights, random_state=42)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": rf_model, 
        "Decision Tree": DecisionTreeClassifier(), 
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    print("\n--- MODEL EĞİTİMİ BAŞLIYOR ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name}: Başarı Oranı: %{acc*100:.2f}")
        
        if name == "Random Forest":
            print(f"\n--- {name} Detaylı Rapor ---")
            print(classification_report(y_test, y_pred))
            # Confusion Matrix grafiği
            plt.figure(figsize=(8,6))
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                        xticklabels=model.classes_, yticklabels=model.classes_)
            plt.title(f'{name} Confusion Matrix')
            plt.ylabel('Gerçek')
            plt.xlabel('Tahmin')
            plt.show()

    # 7. MODEL KAYDETME (Web Arayüzü İçin)
    print("\n💾 SİSTEM GÜNCELLENİYOR...")
    try:
        # Arayüz 'spam_model_rf.pkl' beklediği için ismini böyle yapıyoruz
        joblib.dump(rf_model, 'spam_model_rf.pkl') 
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        print(f"✅ BAŞARILI: Model güncellendi -> 'spam_model_rf.pkl'")
        print(f"✅ BAŞARILI: Vektörleştirici güncellendi -> 'tfidf_vectorizer.pkl'")
    except Exception as e:
        print(f"⚠️ Model kaydedilirken hata oluştu: {e}")

if __name__ == "__main__":
    run_training_pipeline()