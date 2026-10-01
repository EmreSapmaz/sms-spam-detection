import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

def process_clustering():
    # 1. Temizlenmiş veriyi yükle
    df = pd.read_csv('data/spam_cleaned.csv')
    
    # Sadece SPAM mesajlarını alıyoruz 
    spam_df = df[df['label'] == 'spam'].copy()
    print(f"Toplam Spam Mesaj Sayısı: {len(spam_df)}")
    
    # 2. Vektörleştirme 
    # Stop words='english' ile 'the', 'is' gibi kelimeleri atıyoruz.
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    X = vectorizer.fit_transform(spam_df['message'])
    
    # 3. K-Means Modelleme 
    # 4 Kümeye ayırmayı deneyelim (Reklam, Oltalama, Finansal, +1 Diğer)
    k = 4
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    
    # Kümeleri veriye ekle
    spam_df['cluster'] = kmeans.labels_
    
    # 4. Performans Ölçümü: Silhouette Score 
    # 1'e ne kadar yakınsa ayrım o kadar iyi demektir.
    score = silhouette_score(X, kmeans.labels_)
    print(f"Silhouette Skoru: {score:.3f}")
    
    # 5. Kümelerin İçeriğini Anlama 
    print("\n--- KÜME ANALİZİ (Cluster Analysis) ---")
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    cluster_keywords = {}
    
    for i in range(k):
        print(f"\nCluster {i} (Mesaj Sayısı: {len(spam_df[spam_df['cluster'] == i])}):")
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Anahtar Kelimeler: {', '.join(top_terms)}")
        
        # Örnek 3 mesaj gösterelim
        print("Örnek Mesajlar:")
        print(spam_df[spam_df['cluster'] == i]['message'].head(3).values)
        cluster_keywords[i] = top_terms

    # 6. Sonuçları Kaydetme
    # Ana veri setine cluster bilgisini birleştiriyoruz (Ham mesajların cluster'ı -1 olsun)
    df['cluster'] = -1
    df.loc[df['label'] == 'spam', 'cluster'] = spam_df['cluster']
    
    df.to_csv('data/spam_clustered.csv', index=False)
    print("\nKümeleme tamamlandı. 'data/spam_clustered.csv' kaydedildi.")
    
    return spam_df

if __name__ == "__main__":
    process_clustering()