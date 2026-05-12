import streamlit as st
import base64
from PIL import Image
import base64
from io import BytesIO
from model import load_model_5, load_model_10, predict_image
import hashlib

def file_hash(file):
    return hashlib.md5(file.getvalue()).hexdigest()

@st.cache_resource
def get_model_10():
    return load_model_10()

@st.cache_resource
def get_model_5():
    return load_model_5()

def halaman_uji(judul, get_model_func, key_prefix):

    image_key = f"image_{key_prefix}"
    run_key = f"run_test_{key_prefix}"
    hash_key = f"last_hash_{key_prefix}"

    # INIT STATE
    if image_key not in st.session_state:
        st.session_state[image_key] = None
    if run_key not in st.session_state:
        st.session_state[run_key] = False
    if hash_key not in st.session_state:
        st.session_state[hash_key] = None

    # TITLE
    st.markdown(f'<div class="test-title">Halaman Pengujian {judul}</div>', unsafe_allow_html=True)
    st.markdown('<div class="test-subtitle">Upload atau ambil foto daun tomat untuk menguji klasifikasi penyakit.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="test-card">
        <div style="text-align: center; font-weight: bold;">
            Cara Menggunakan Sistem Ini
        </div>
        <ol style="font-size:0.7rem;">
            <li>Pilih <b>Browse Files</b> atau <b>Take Photo</b></li>
            <li>Masukkan gambar daun tomat atau foto gambar daun tomat dengan maksimal limit 1MB</li>
            <li>Tekan tombol <b>Uji</b> untuk melihat klasifikasi penyakit daun tomat yang di inginkan</li>
            <li>Agar hasil lebih akurat, usahakan foto daun tomat dengan background abu - abu</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # LAYOUT
    col1, col2, col3 = st.columns([3,1,3])

    # INPUT
    with col1:
        with st.container():
            st.markdown('<div class="card-input">', unsafe_allow_html=True)

            st.markdown("<div style='text-align:center; font-weight:700;'>Input Gambar</div>", unsafe_allow_html=True)

            preview = st.empty()
            st.markdown("<div style='margin-top:10px;'></div>", unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "📂 Upload gambar",
                key=f"upload_{key_prefix}",
                label_visibility="collapsed",
                type=["jpg", "jpeg", "png"]
            )


            MAX_SIZE = 1 * 1024 * 1024

            if uploaded_file is not None:
                if uploaded_file.size > MAX_SIZE:
                    st.error("Ukuran gambar maksimal 1 MB!")
                    st.session_state[image_key] = None
                else:
                    new_hash = file_hash(uploaded_file)

                    if new_hash != st.session_state[hash_key]:
                        st.session_state[image_key] = Image.open(uploaded_file)
                        st.session_state[run_key] = False
                        st.session_state[hash_key] = new_hash

            if st.session_state[image_key] is None:
                preview.markdown("""
                <div style="
                    height:220px;
                    border:2px dashed #cbd5e1;
                    border-radius:12px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    margin:5px 0;
                    background-color:#f8fafc;
                    overflow:hidden;
                ">
                    <div style="color:#64748b; font-size:0.8rem;">
                        Belum ada gambar dipilih
                    </div>
                """, unsafe_allow_html=True)
            else:
                preview.markdown(f"""
                <div style="
                    height:220px;
                    border-radius:12px;
                    margin:5px 0;
                    background-color:#f8fafc;
                    overflow:hidden;
                ">
                    <img src="data:image/png;base64,{image_to_base64(st.session_state[image_key])}"
                    style="
                        width:100%;
                        height:100%;
                        object-fit:cover;
                        display:block;
                    ">
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        is_ready = st.session_state.get(image_key) is not None

        wrapper = st.container()

        st.markdown("""
        <style>
        div[data-testid="stVerticalBlock"] {
            overflow: visible !important;
        }
        </style>
        """, unsafe_allow_html=True)

        with wrapper:
            st.markdown("""
            <style>
            .spacer { height: 90px; }
            @media (max-width: 768px) {
                .spacer { height: 0.5px; }
            }
            </style>
            <div class="spacer"></div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="arrow">→</div>', unsafe_allow_html=True)

            if st.button(
                "Uji",
                key=f"btn_uji_{key_prefix}",
                disabled=(not is_ready) or st.session_state.get(run_key, False),
                type="secondary"
            ):
                st.session_state[run_key] = True
                st.rerun()

    # HASIL
    with col3:
        st.markdown('<div class="card-input">', unsafe_allow_html=True)

        if st.session_state[run_key] and st.session_state[image_key] is not None:

            model, class_names, device = get_model_func()
            probs = predict_image(model, st.session_state[image_key], device)

            data = list(zip(class_names, probs * 100))
            data = sorted(data, key=lambda x: x[1], reverse=True)

            top_label, top_value = data[0]

            html = ""
            for i, (label, value) in enumerate(data, 1):
                clean_label = format_label(label)
                html += f"""<div style="margin-bottom:10px;">
                <div style="display:flex; justify-content:space-between; font-size:0.75rem;">
                <div>{i}. {clean_label}</div>
                <div>{value:.2f}%</div>
                </div>
                <div style="width:100%;height:6px;background:#e5e7eb;border-radius:10px;margin-top:4px;">
                <div style="width:{value}%;height:100%;background:#2563eb;border-radius:10px;"></div>
                </div>
                </div>"""

            top_label_clean = format_label(top_label)

            st.markdown(f"""
            <div class="test-card">
                <div style="text-align: center; font-weight: bold;">
                    Hasil Klasifikasi
                </div>
                <div class="result-card">
                    <div class="result-title">Prediksi Teratas:</div>
                    <div class="result-main">{top_label_clean}</div>
                    <div class="result-percent">{top_value:.2f}%</div>
                </div>
                <div style="font-size:0.7rem;">
                    <b>Ranking Semua Prediksi:</b><br>
                <div class="scroll-box">{html}</div>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="test-card">
                <div style="text-align:center; font-size:0.8rem;">
                    Belum ada hasil
                </div>
            </div>
            """, unsafe_allow_html=True)

def format_label(label):
    label = label.replace("Tomato___", "")
    label = label.replace("_", " ")
    label = label.lower()

    words = [w for w in label.split() if w != "tomato"]
    label = " ".join(words)

    mapping = {
        "Spider Mites Two Spotted Spider Mite": "Spider Mite"
    }

    return mapping.get(label.title(), label.title())

# CONFIG
st.set_page_config(
    page_title="Menu Sidebar",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
    
img1 = get_base64("Foto/foto_diri.png")
img2 = get_base64("Foto/foto_dosen.png")

st.markdown("""
<style>

.stApp {
    background-color: #ffffff;
    color: #000000 !important;
}

.main .block-container {
    padding-top: 2rem;
    padding-bottom: 0rem;
    max-width: 1100px;
}

section[data-testid="stSidebar"] {
    background-color: #0F172A;
}

.menu-title {
    text-align: center;
    color: white;
    font-size: 1.2rem;
    font-weight: 600;
    margin: 0 0 12px 0;
}

.sidebar-divider {
    height: 1px;
    background: #1E293B;
    margin: 2px -16px 4px -16px;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
    margin-top: 0 !important;
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] {
    padding: 0 12px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 12px 20px !important;
    border-radius: 6px;
    color: #e2e8f0 !important;
    margin: 4px 0;
    width: 100%;
    box-sizing: border-box;
}

section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {
    display: none;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 0.7rem !important;
    margin: 0 !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #1E3A5F;
}

section[data-testid="stSidebar"] label:has(input:checked) {
    background-color: #1E3A5F !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
            
header[data-testid="stHeader"]::after {
    display: none !important;
}

/* LANDING PAGE */
.main-title {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 20px;
}

.main-card {
    background: #ffffff;
    padding: 20px 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e2e8f0;
}

.feature-card {
    background: #ffffff;
    padding: 16px;
    border-radius: 10px;
    text-align: center;
    border: 1px solid #e2e8f0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    height: 120px;
    margin-bottom: 10px;
}

.feature-title {
    font-weight: 700;
    font-size: 1rem;
}

.feature-text {
    font-size: 0.7rem;
    color: #475569;
}
            
/* TEST PAGE */
.test-title {
    text-align: center;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 5px;
}

.test-subtitle {
    text-align: center;
    font-size: 0.75rem;
    color: #64748b;
    margin-bottom: 15px;
}

.test-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 10px 10px 10px 10px;
    border: 1px solid #e2e8f0;
}

.center-box {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 100%;
}

.result-card {
    background: #f1f5f9;
    border-radius: 10px;
    padding: 10px;
    border: 1px solid #e2e8f0;
    margin: 10px 0;
}

.result-title {
    font-size: 0.7rem;
    color: #64748b;
}

.result-main {
    font-size: 1rem;
    font-weight: 700;
}

.result-percent {
    font-size: 0.7rem;
    color: #64748b;
}
            
button[kind="secondary"] {
    width: auto !important;
    background-color: #2563eb !important;
    text-align: center !important;
    color: white !important;
    border: none !important;
    border-radius: 10px;
    margin: auto !important;
}

button[kind="secondary"]:hover {
    background-color: #1d4ed8 !important;
}

button[kind="secondary"]:disabled {
    text-align: center !important;
    background-color: #9ca3af !important;
    color: white !important;
    cursor: not-allowed;
    opacity: 1 !important;
}
            
div[data-testid="stVerticalBlock"] {
    display: flex;
    flex-direction: column;
    align-items: center;
}
        
.arrow {
    text-align: center;
    font-size: 30px;
    margin-bottom: 10px;
}

/* ===== BOX UPLOAD ===== */
[data-testid="stFileUploader"] section {
    background-color: #f1f1f1 !important;
    padding: 10px !important;
    border-radius: 12px !important;

    min-height: auto !important;
    height: auto !important;
}

/* ===== HILANGKAN AREA DRAG BESAR ===== */
[data-testid="stFileUploaderDropzone"] {
    padding: 0 !important;
    min-height: auto !important;
}

/* ===== TOMBOL ===== */
[data-testid="stFileUploader"] button {
    background-color: #1a73e8 !important;
    color: white !important;
    font-size: 0.85rem !important;
    border-radius: 10px !important;
    border: none !important;

    padding: 10px 18px !important;
    height: auto !important;
}

/* ===== TEXT ===== */
[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small {
    font-size: 0.65rem !important;
    color: black !important;
}

.scroll-box {
    max-height: 220px;   
    overflow-y: auto;
    margin-top: 8px;
    padding-right: 6px;
}

.scroll-box::-webkit-scrollbar {
    width: 6px;
}

.scroll-box::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 10px;
}

.scroll-box::-webkit-scrollbar-track {
    background: transparent;
}
            
[data-testid="stFileUploaderFile"] {
    display: none;
}

[data-testid="stFileUploader"] > div:nth-child(3) {
    display: none;
}
            
/* ABOUT PAGE */
.about-title {
    text-align: center;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 10px;
}

[data-testid="stImage"] img {
    width: 150px !important;
    height: 150px !important;
    border-radius: 50% !important;
    object-fit: cover !important;
    display: block;
}

/* CENTER TEXT */
.profile-text {
    text-align: center;
}

.profile-name {
    font-size: 1.2rem;
    font-weight: 700;
}

.profile-role {
    font-size: 0.95rem;
    color: #555;
}           

.about-card {
    max-width: 650px;
    margin: 0 auto;
    background: #ffffff;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    text-align: center;
}
        
</style>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div class="menu-title">Menu</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    with st.container():
        menu = st.radio("", [
            "🏠 Halaman Utama",
            "🔬 Pengujian 10 kelas",
            "🔬 Pengujian 5 kelas",
            "ℹ️ Informasi Tentang Pembuat"
        ], index=0)

# HALAMAN UTAMA
if menu == "🏠 Halaman Utama":
    st.markdown("""
    <div class="main-title">
    Optimisasi Hyperparameter Vision Transformer Untuk<br>
    Klasifikasi Penyakit Daun Tomat
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="main-card">
        <h5>Tentang Website Ini</h5>
        <p style="font-size:0.7rem;">
        Website ini dibuat untuk menguji apakah performa Vision Transformer yang sudah dioptimisasi
        dapat mengklasifikasikan penyakit daun tomat dengan baik.
        </p>
        <p style="font-size:0.7rem;">
        Vision Transformer (ViT) menggunakan mekanisme attention untuk memproses gambar.
        Optimisasi hyperparameter dilakukan untuk meningkatkan akurasi klasifikasi.
        </p>
        <p style="font-size:0.7rem;">
        Gunakan halaman Pengujian untuk mencoba mengunggah atau mengambil foto daun tomat dan lihat hasil klasifikasi penyakit secara real-time menggunakan model yang telah dioptimisasi.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Vision Transformer</div>
            <div class="feature-text">
            Arsitektur modern untuk klasifikasi citra berbasis attention.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Optimisasi Hyperparameter</div>
            <div class="feature-text">
            Meningkatkan performa model secara signifikan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-title">Klasifikasi Real-time</div>
            <div class="feature-text">
            Prediksi langsung dari gambar daun tomat.
            </div>
        </div>
        """, unsafe_allow_html=True)

elif menu == "🔬 Pengujian 10 kelas":
    halaman_uji("10 kelas", get_model_10, "10")

elif menu == "🔬 Pengujian 5 kelas":
    halaman_uji("5 kelas", get_model_5, "5")

elif menu == "ℹ️ Informasi Tentang Pembuat":

    st.markdown('<div class="about-title">Info tentang Pembuat</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="profile-image">', unsafe_allow_html=True)
        st.image("Foto/foto_diri.png")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="profile-text">
            <div class="profile-name">
                Christian Fernando
            </div>
            <div class="profile-role">
                Mahasiswa Teknik Informatika 
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="profile-image">', unsafe_allow_html=True)
        st.image("Foto/foto_dosen.png")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="profile-text">
            <div class="profile-name">
                Dra. Chairisni Lubis, M.Kom.
            </div>
            <div class="profile-role">
                Dosen Pembimbing Utama
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="about-card">
        <p style="font-size:0.7rem;">
        Saya merupakan mahasiswa Program Studi Teknik Informatika angkatan 2022 di Universitas Tarumanagara. 
        Melalui rancangan ini, saya bersama dosen pembimbing mengembangkan sebuah aplikasi website berbasis 
        Optimisasi Hyperparameter Vision Transformer (ViT) untuk mengklasifikasikan penyakit pada daun tomat secara otomatis. 
        Aplikasi ini dirancang untuk membantu proses identifikasi penyakit daun tomat dengan lebih cepat dan akurat 
        melalui pengunggahan gambar maupun pengambilan foto secara real-time. 
        Pengembangan website ini diharapkan dapat menjadi salah satu penerapan teknologi kecerdasan buatan 
        yang bermanfaat dalam mendukung pemantauan kesehatan tanaman tomat.
        </p>
    </div>
    """, unsafe_allow_html=True)
    