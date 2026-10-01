import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Ayarlar
nltk.download('stopwords', quiet=True)
save_folder = 'rapor_dosyalari'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f"📁 '{save_folder}' klasörü oluşturuldu.")

def clean_text(text):
    ps = PorterStemmer()
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    all_stopwords = stopwords.words('english')
    words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(words)

def generate_all_assets():
    print("--- 1. VERİ HAZIRLANIYOR ---")
    
    # Veri Birleştirme (Standart Prosedür)
    df_raw = pd.read_csv('data/spam_cleaned.csv')
    df_ham = df_raw[df_raw['label'] == 'ham'].copy()
    df_ham['target'] = 'Normal'
    
    df_clustered = pd.read_csv('data/spam_clustered.csv')
    df_spam = df_clustered[df_clustered['label'] == 'spam'].copy()
    label_map = {0: 'Abonelik', 1: 'Dolandiricilik', 2: 'Bahis', 3: 'Promosyon'}
    df_spam['target'] = df_spam['cluster'].map(label_map)
    
    df_final = pd.concat([df_ham[['message', 'target']], df_spam[['message', 'target']]])
    
    # Active Learning verilerini de katalım (Eğer varsa)
    if os.path.exists('data/feedback_log.csv'):
        try:
            df_fb = pd.read_csv('data/feedback_log.csv')
            df_fb.rename(columns={'Message': 'message', 'True_Label': 'target'}, inplace=True)
            df_final = pd.concat([df_final, df_fb[['message', 'target']]])
            print("   -> Geri bildirim verileri rapora dahil edildi.")
        except:
            pass

    df_final.dropna(subset=['target'], inplace=True)
    
    # --- GÖRSEL 1: VERİ SETİ İSTATİSTİĞİ (Pasta Grafik) ---
    print("--- 2. VERİ İSTATİSTİKLERİ ÇİZİLİYOR ---")
    plt.figure(figsize=(10, 6))
    counts = df_final['target'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Veri Seti Sınıf Dağılımı')
    plt.savefig(f'{save_folder}/1_Veri_Dagilimi_Pasta.png')
    plt.close()
    
    # Sütun Grafik Hali (Sayıları görmek için)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title('Sınıf Başına Mesaj Sayısı')
    plt.xticks(rotation=45)
    plt.savefig(f'{save_folder}/1_Veri_Dagilimi_Bar.png')
    plt.close()

    # --- MODEL EĞİTİMİ ---
    print("--- 3. MODELLER EĞİTİLİYOR VE RAPORLANIYOR ---")
    df_final['clean_text'] = df_final['message'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df_final['clean_text']).toarray()
    y = df_final['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    accuracies = {}
    
    for name, model in models.items():
        print(f"   -> {name} analiz ediliyor...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # 1. Accuracy Kaydet
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc * 100
        
        # 2. Detaylı Raporu (Precision, Recall, F1) TXT olarak kaydet
        report = classification_report(y_test, y_pred)
        with open(f'{save_folder}/Rapor_{name.replace(" ", "_")}.txt', 'w') as f:
            f.write(f"MODEL: {name}\n")
            f.write(f"GENEL BAŞARI (ACCURACY): %{acc*100:.2f}\n")
            f.write("-" * 60 + "\n")
            f.write(report)
            
        # 3. Confusion Matrix (Resim)
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'{name} - Hata Matrisi')
        plt.ylabel('Gerçek')
        plt.xlabel('Tahmin')
        plt.tight_layout()
        plt.savefig(f'{save_folder}/CM_{name.replace(" ", "_")}.png')
        plt.close()

    # --- GÖRSEL 2: MODEL KIYASLAMA (Sütun Grafik) ---
    print("--- 4. KIYASLAMA GRAFİĞİ ÇİZİLİYOR ---")
    plt.figure(figsize=(10, 6))
    # Sonuçları sıralayalım
    sorted_acc = dict(sorted(accuracies.items(), key=lambda item: item[1], reverse=True))
    
    sns.barplot(x=list(sorted_acc.keys()), y=list(sorted_acc.values()), palette='magma')
    plt.ylim(80, 100)  # Grafiği 80-100 arasına odakla ki farklar görünsün
    plt.title('Modellerin Başarı Karşılaştırması (%)')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.xticks(rotation=15)
    
    # Barların üstüne sayıları yazalım
    for i, v in enumerate(sorted_acc.values()):
        plt.text(i, v + 0.2, f"%{v:.1f}", ha='center', fontweight='bold')
        
    plt.savefig(f'{save_folder}/2_Model_Kiyaslama.png')
    plt.close()
    
    print(f"\n✅ İŞLEM TAMAMLANDI! Tüm dosyalar '{save_folder}' klasörüne kaydedildi.")

if __name__ == "__main__":
    generate_all_assets()