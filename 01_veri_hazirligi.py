import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas ayarları (Çıktıyı düzgün görmek için)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def load_and_clean_data(file_path):
    print("--- Veri Yükleniyor ---")
    # Kaggle veri seti genelde 'latin-1' formatındadır, utf-8 hata verebilir.
    df = pd.read_csv(file_path, encoding='latin-1')
    
    # 1. Gereksiz Sütunları Temizleme (Veri Setine özel durum)
    # Genelde 'Unnamed: 2', 'Unnamed: 3' gibi boş sütunlar gelir, onları atıyoruz.
    df = df[['v1', 'v2']]
    
    # 2. Sütun İsimlerini Anlaşılır Yapma
    df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
    
    print(f"İlk Veri Boyutu: {df.shape}")
    
    # 3. Tekrar Eden Verileri (Duplicates) Kaldırma
    
    duplicates = df.duplicated().sum()
    print(f"Tekrar eden veri sayısı: {duplicates}")
    df.drop_duplicates(inplace=True)
    
    # 4. Eksik Veri Kontrolü 
    print("\nEksik Veri Durumu:")
    print(df.isnull().sum())
    
    # 5. Yeni Özellik: Mesaj Uzunluğu 
    # Uzunluk spam ayrımında önemli bir ipucu olabilir.
    df['length'] = df['message'].apply(len)
    
    print(f"\nTemizlik Sonrası Veri Boyutu: {df.shape}")
    return df

def visualize_data_distribution(df):
    # Hafta 3: Histogram ve Dağılım Grafikleri
    plt.figure(figsize=(12, 6))
    
    # Ham vs Spam sayısal dağılımı
    plt.subplot(1, 2, 1)
    sns.countplot(x='label', data=df)
    plt.title('Spam vs Ham Dağılımı')
    
    # Mesaj uzunluklarının dağılımı (Histogram)
    plt.subplot(1, 2, 2)
    sns.histplot(df[df['label'] == 'ham']['length'], bins=50, alpha=0.5, label='Ham', color='blue')
    sns.histplot(df[df['label'] == 'spam']['length'], bins=50, alpha=0.5, label='Spam', color='red')
    plt.title('Mesaj Uzunluk Dağılımı')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# --- ANA AKIŞ (Düzeltilmiş) ---
if __name__ == "__main__":
    # Dosya yolunu kendine göre düzenle
    file_path = 'data/spam.csv' 
    
    try:
        df = load_and_clean_data(file_path)
        
        # İlk 5 satırı görelim
        print("\n--- Veriden Örnekler ---")
        print(df.head())

        # --- DÜZELTME BURADA: ÖNCE KAYDET, SONRA GRAFİK ÇİZ ---
        # Temizlenmiş veriyi K-Means için kayedelim
        df.to_csv('data/spam_cleaned.csv', index=False)
        print("\n✅ BAŞARILI: Temizlenmiş veri 'data/spam_cleaned.csv' olarak kaydedildi.")
        
        # Grafikleri çizdirelim (Hata verirse bile dosya kaydedilmiş olacak)
        print("Grafikler oluşturuluyor...")
        visualize_data_distribution(df)
        
    except FileNotFoundError:
        print("HATA: 'data/spam.csv' dosyası bulunamadı. Lütfen dosya yolunu kontrol et.")
    except Exception as e:
        print(f"Beklenmedik bir hata oluştu ama dosyanız kaydedilmiş olabilir: {e}")