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
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

# Ayarlar
nltk.download('stopwords', quiet=True)
save_folder = 'rapor_dosyalari'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

def clean_text(text):
    ps = PorterStemmer()
    text = str(text).lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    all_stopwords = stopwords.words('english')
    words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(words)

def generate_advanced_visuals():
    print("--- VERİ HAZIRLANIYOR ---")
    
    # Veri Yükleme
    df_raw = pd.read_csv('data/spam_cleaned.csv')
    df_ham = df_raw[df_raw['label'] == 'ham'].copy()
    df_ham['target'] = 'Normal'
    
    df_clustered = pd.read_csv('data/spam_clustered.csv')
    df_spam = df_clustered[df_clustered['label'] == 'spam'].copy()
    label_map = {0: 'Abonelik', 1: 'Dolandiricilik', 2: 'Bahis', 3: 'Promosyon'}
    df_spam['target'] = df_spam['cluster'].map(label_map)
    
    df_final = pd.concat([df_ham[['message', 'target']], df_spam[['message', 'target']]])
    df_final.dropna(subset=['target'], inplace=True)
    
    # Active Learning verisi varsa ekle
    if os.path.exists('data/feedback_log.csv'):
        try:
            df_fb = pd.read_csv('data/feedback_log.csv')
            df_fb.rename(columns={'Message': 'message', 'True_Label': 'target'}, inplace=True)
            df_final = pd.concat([df_final, df_fb[['message', 'target']]])
        except:
            pass

    # Vektörleştirme
    print("--- METİN İŞLENİYOR ---")
    df_final['clean_text'] = df_final['message'].apply(clean_text)
    tfidf = TfidfVectorizer(max_features=3000)
    X = tfidf.fit_transform(df_final['clean_text']).toarray()
    y = df_final['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Log. Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "Decision Tree": DecisionTreeClassifier(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    # Metrikleri Saklamak İçin Sözlük
    metrics_data = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1-Score': []
    }
    
    # Korelasyon için tahminleri sakla
    model_predictions = pd.DataFrame()
    model_predictions['Gerçek'] = y_test

    print("--- MODELLER ANALİZ EDİLİYOR ---")
    
    for name, model in models.items():
        print(f"   -> {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Tahminleri kaydet (Korelasyon için)
        model_predictions[name] = y_pred
        
        # Metrik Hesapla (Weighted Average kullanıyoruz çünkü dengesiz veri)
        acc = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        metrics_data['Model'].append(name)
        metrics_data['Accuracy'].append(acc)
        metrics_data['Precision'].append(precision)
        metrics_data['Recall'].append(recall)
        metrics_data['F1-Score'].append(f1)
        
        # --- GÖRSEL 1: SINIFLANDIRMA RAPORU ISI HARİTASI (Visual Classification Report) ---
        # Bu, metin raporun görsel halidir.
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).iloc[:-1, :].T # Support satırını atıp transpoze alıyoruz
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.2f', vmin=0, vmax=1)
        plt.title(f'{name} - Detaylı Performans Analizi')
        plt.savefig(f'{save_folder}/Heatmap_Rapor_{name.replace(" ", "_")}.png')
        plt.close()

    # --- GÖRSEL 2: TÜM METRİKLERİN KIYASLAMASI (Grouped Bar Chart) ---
    print("--- KIYASLAMA GRAFİKLERİ ÇİZİLİYOR ---")
    df_metrics = pd.DataFrame(metrics_data)
    # Veriyi grafik formatına çevir (Melt)
    df_melted = df_metrics.melt(id_vars="Model", var_name="Metrik", value_name="Skor")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted, x='Model', y='Skor', hue='Metrik', palette='viridis')
    plt.title('Tüm Modellerin Detaylı Karşılaştırması (Accuracy, Precision, Recall, F1)', fontsize=14)
    plt.ylim(0.8, 1.0) # Farkları görmek için alt sınırı 0.8 yapıyoruz
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{save_folder}/3_Detayli_Kiyaslama_Grafigi.png')
    plt.close()

    # --- GÖRSEL 3: MODEL KORELASYON MATRİSİ ---
    # Modellerin tahminlerinin birbirine ne kadar benzediğini gösterir.
    # Kategorik veriyi sayıya çevirip korelasyon bakacağız.
    print("--- KORELASYON MATRİSİ ÇİZİLİYOR ---")
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    df_corr = model_predictions.drop(columns=['Gerçek']).apply(le.fit_transform)
    corr_matrix = df_corr.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Modeller Arası Tahmin Korelasyonu\n(Hangi modeller benzer kararlar veriyor?)')
    plt.savefig(f'{save_folder}/4_Model_Korelasyon_Matrisi.png')
    plt.close()
    
    print(f"\n✅ HARİKA! Tüm yeni görseller '{save_folder}' klasörüne eklendi.")
    print("1. 'Heatmap_Rapor_...' -> Her modelin detaylı renkli raporu.")
    print("2. '3_Detayli_Kiyaslama_Grafigi.png' -> F1, Recall, Precision yan yana.")
    print("3. '4_Model_Korelasyon_Matrisi.png' -> Modellerin benzerlik haritası.")

if __name__ == "__main__":
    generate_advanced_visuals()