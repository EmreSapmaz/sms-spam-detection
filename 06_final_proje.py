import streamlit as st
import joblib
import re
import pandas as pd
import os
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Sayfa Ayarları
st.set_page_config(page_title="SMS Spam Dedektörü Pro", page_icon="🛡️", layout="wide")

# NLTK İndirmeleri
nltk.download('stopwords', quiet=True)

# --- FONKSİYONLAR ---
def clean_text(text):
    ps = PorterStemmer()
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    words = text.split()
    all_stopwords = stopwords.words('english')
    words = [ps.stem(word) for word in words if word not in all_stopwords]
    return ' '.join(words)

def generate_wordcloud(category):
    """Seçilen kategoriye ait kelime bulutu oluşturur"""
    try:
        # Önceden hazırladığımız kümelenmiş veriyi okuyalım
        df = pd.read_csv('data/spam_clustered.csv')
        
        # Etiket haritasını tekrar kuralım 
        label_map = {0: 'Abonelik_Hizmet', 1: 'Dolandiricilik_Odul', 2: 'Bahis_Yarisma', 3: 'Cihaz_Promosyon'}
        df['label_name'] = df['cluster'].map(label_map)
        
        # Sadece ilgili kategorideki mesajları al
        text_data = " ".join(df[df['label_name'] == category]['message'].astype(str))
        
        if not text_data:
            return None
            
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text_data)
        return wordcloud
    except Exception as e:
        return None

def save_feedback(message, true_label, predicted_label):
    """Kullanıcı geri bildirimini kaydeder"""
    file_path = 'data/feedback_log.csv'
    new_data = pd.DataFrame([[message, true_label, predicted_label]], columns=['Message', 'True_Label', 'Predicted_Label'])
    
    if not os.path.exists(file_path):
        new_data.to_csv(file_path, index=False)
    else:
        new_data.to_csv(file_path, mode='a', header=False, index=False)

def predict_message(model, tfidf, message):
    cleaned_msg = clean_text(message)
    vectorized_msg = tfidf.transform([cleaned_msg]).toarray()
    
    probs = model.predict_proba(vectorized_msg)[0]
    classes = model.classes_
    
    max_prob_idx = probs.argmax()
    prediction = classes[max_prob_idx]
    confidence = probs[max_prob_idx] * 100
    
    risk_msg = ""
    # Risk Kontrolü (Hibrit Yapı)
    if prediction == 'Normal' and confidence < 80:
        spam_probs = {c: p for c, p in zip(classes, probs) if c != 'Normal'}
        best_spam_class = max(spam_probs, key=spam_probs.get)
        spam_conf = spam_probs[best_spam_class] * 100
        
        if spam_conf > 15:
            risk_msg = f"⚠️ Model 'Normal' dedi ama Risk Algoritması **{best_spam_class}** tespit etti. (Risk: %{spam_conf:.1f})"
            prediction = best_spam_class
            confidence = spam_conf

    return prediction, confidence, risk_msg

# --- ARAYÜZ ---
st.title("🛡️ Akıllı SMS Güvenlik Duvarı")
st.markdown("**Makine Öğrenmesi & Doğal Dil İşleme (NLP) Dönem Projesi**")
st.markdown("---")

# Sol Panel  ve Sağ Panel 
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📩 Mesaj Analizi")
    
    # Modeli Yükle
    try:
        model = joblib.load('spam_model_rf.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        st.success("✅ Sistem Aktif")
    except:
        st.error("Model dosyaları bulunamadı! Lütfen eğitimi tekrar yapın.")
        st.stop()

    user_input = st.text_area("Şüpheli SMS'i buraya yapıştırın:", height=150, placeholder="Örn: URGENT! You have won £1000 prize...")
    
    analyze_btn = st.button("🔍 ANALİZ ET", use_container_width=True)

# Sonuçları Saklamak için Session State
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None

if analyze_btn and user_input:
    with st.spinner('Yapay Zeka düşünüyor...'):
        pred, conf, risk = predict_message(model, tfidf, user_input)
        st.session_state['prediction'] = pred
        st.session_state['conf'] = conf
        st.session_state['risk'] = risk
        st.session_state['msg'] = user_input

# Analiz Sonucu Gösterimi
if st.session_state['prediction']:
    pred = st.session_state['prediction']
    conf = st.session_state['conf']
    risk = st.session_state['risk']
    
    with col1:
        st.divider()
        if pred == "Normal":
            st.success(f"### ✅ GÜVENLİ (Normal)")
            st.progress(conf/100)
            st.write(f"Güven Skoru: **%{conf:.2f}**")
        else:
            st.error(f"### 🚨 SPAM TESPİT EDİLDİ")
            st.warning(f"Kategori: **{pred}**")
            st.progress(conf/100)
            st.write(f"Güven Skoru: **%{conf:.2f}**")
            
            if risk:
                st.caption(risk)
        
        # --- GERİ BİLDİRİM SİSTEMİ (Active Learning) ---
        st.divider()
        st.write("📝 **Sistem Yanıldı mı? Bize Öğretin:**")
        
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            if st.button("👍 Doğru Bildin"):
                st.toast("Geri bildiriminiz için teşekkürler! Model bunu öğrendi.")
                # Burada 'Doğru' olduğunu da kaydedebiliriz istersen
        
        with fb_col2:
            if st.button("👎 Yanlış Bildin"):
                st.session_state['show_feedback_form'] = True

        if st.session_state.get('show_feedback_form'):
            correct_label = st.selectbox("Aslında ne olmalıydı?", 
                                       ['Normal', 'Dolandiricilik_Odul', 'Bahis_Yarisma', 'Cihaz_Promosyon', 'Abonelik_Hizmet'])
            if st.button("Düzeltmeyi Gönder"):
                save_feedback(st.session_state['msg'], correct_label, pred)
                st.success("✅ Veri tabanına kaydedildi! Bir sonraki eğitimde model bunu düzeltecek.")
                st.session_state['show_feedback_form'] = False

    # --- SAĞ PANEL: WORD CLOUD ---
    with col2:
        if pred != "Normal":
            st.subheader(f"☁️ '{pred}' Kategorisi Analizi")
            st.info("Bu kategorideki spam mesajlarda en sık geçen kelimeler:")
            
            wc = generate_wordcloud(pred)
            if wc:
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis("off")
                st.pyplot(plt)
            else:
                st.write("Görselleştirme için yeterli veri yok.")
        else:
            st.subheader("📊 Güvenli İletişim")
            st.image("https://img.freepik.com/free-vector/safe-concept-illustration_114360-1538.jpg?w=740", caption="Mesajınız temiz görünüyor.", width=400)

# Alt Bilgi
st.markdown("---")
st.caption("Geliştiriciler: Emre & Halil | Model: Weighted Random Forest + NLP | Ders: Makine Öğrenmesi")