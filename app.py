"""
Hotel LuxStay - Concierge Virtual
Interface moderna estilo chat com RAG, Transcrição, Tradução e Áudio.
"""

import os
import tempfile
import streamlit as st
from groq import Groq
from gtts import gTTS
import whisper
from deep_translator import GoogleTranslator
from dotenv import load_dotenv
import rag

load_dotenv()

# ---------- Configuração da página ----------
st.set_page_config(
    page_title="Hotel LuxStay - Concierge Virtual",
    page_icon="🏨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- CSS customizado ----------
st.markdown("""
<style>
/* Fundo e tema geral */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #0f1724 0%, #1a2332 100%);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #1a2332;
    border-right: 1px solid #2d3748;
}

/* Header do hotel */
.hotel-header {
    text-align: center;
    padding: 2rem 1rem 1rem;
    background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 50%, #1e3a5f 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #3d6a97;
    box-shadow: 0 4px 20px rgba(30, 58, 95, 0.3);
}

.hotel-header h1 {
    color: #f0c040;
    font-size: 2.2rem;
    margin: 0;
    font-family: 'Georgia', serif;
    letter-spacing: 2px;
}

.hotel-header p {
    color: #a8c8e8;
    font-size: 1rem;
    margin-top: 0.3rem;
}

.hotel-stars {
    color: #f0c040;
    font-size: 1.2rem;
    letter-spacing: 4px;
}

/* Chat container */
.chat-container {
    max-height: 500px;
    overflow-y: auto;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* Bolhas de mensagem */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1rem;
}

.msg-user .bubble {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    padding: 0.8rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 75%;
    font-size: 0.95rem;
    box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3);
}

.msg-bot {
    display: flex;
    justify-content: flex-start;
    margin-bottom: 1rem;
    align-items: flex-start;
    gap: 0.5rem;
}

.msg-bot .avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #f0c040, #d4a030);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}

.msg-bot .bubble {
    background: #1e293b;
    color: #e2e8f0;
    padding: 0.8rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 75%;
    font-size: 0.95rem;
    border: 1px solid #334155;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
}

/* Botões de sugestão */
.suggestion-btn {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #3d5a80;
    border-radius: 20px;
    padding: 0.5rem 1rem;
    color: #93c5fd;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    margin: 0.25rem;
}

.suggestion-btn:hover {
    background: #2d3f5a;
    border-color: #60a5fa;
}

/* Status badge */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 500;
}

.status-online {
    background: #064e3b;
    color: #6ee7b7;
    border: 1px solid #065f46;
}

/* Input styling */
[data-testid="stTextArea"] textarea {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2) !important;
}

/* Botão enviar */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3b82f6, #2563eb) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
}

.stButton > button {
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    background: #1e293b !important;
    color: #93c5fd !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #2d3f5a !important;
    border-color: #60a5fa !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
}

/* Divider */
hr {
    border-color: #2d3748 !important;
}

/* Sidebar items */
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: #a8c8e8 !important;
}

/* Audio player */
audio {
    border-radius: 8px;
}

/* Info/Success/Warning boxes */
.stAlert {
    border-radius: 10px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Header elegante ----------
st.markdown("""
<div class="hotel-header">
    <div class="hotel-stars">★ ★ ★ ★ ★</div>
    <h1>HOTEL LUXSTAY</h1>
    <p>Concierge Virtual • Atendimento Inteligente 24h</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### ⚙️ Configurações")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Obtenha em https://console.groq.com",
    )

    st.markdown("---")

    st.markdown("#### 🌐 Idioma da Resposta")
    idiomas = {
        "Português (Brasil)": "pt",
        "English": "en",
        "Español": "es",
        "Français": "fr",
        "Deutsch": "de",
        "Italiano": "it",
        "日本語 (Japonês)": "ja",
        "中文 (Chinês)": "zh-CN",
    }
    idioma_selecionado = st.selectbox(
        "Traduzir resposta para:",
        options=list(idiomas.keys()),
        index=0,
        label_visibility="collapsed",
    )
    codigo_idioma = idiomas[idioma_selecionado]

    st.markdown("---")

    if st.button("📚 Carregar Base de Conhecimento", use_container_width=True):
        with st.spinner("Carregando..."):
            try:
                rag.carregar_base(status_callback=lambda msg: None)
                st.success("✅ Pronto!")
            except Exception as e:
                st.error(f"Erro: {str(e)}")

    if rag.esta_carregado():
        st.markdown('<span class="status-badge status-online">● Base ativa</span>', unsafe_allow_html=True)
    else:
        st.warning("⚠️ Clique acima para ativar")

    st.markdown("---")
    st.markdown("##### 🛠️ Tecnologias")
    st.markdown("""
    - 🎙️ Whisper (transcrição)  
    - 📚 RAG (busca inteligente)  
    - 🌐 Tradutor (8 idiomas)  
    - 🔊 gTTS (voz)  
    - 🤖 Llama 3.3 (Groq)
    """)


# ---------- Funções auxiliares ----------
def transcrever_audio(arquivo_audio):
    """Transcreve áudio usando Whisper."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=os.path.splitext(arquivo_audio.name)[1]
    ) as tmp:
        tmp.write(arquivo_audio.read())
        tmp_path = tmp.name

    model = whisper.load_model("base")
    resultado = model.transcribe(tmp_path)
    os.unlink(tmp_path)
    return resultado["text"]


def traduzir_texto(texto, idioma_destino):
    """Traduz texto para o idioma selecionado."""
    if idioma_destino == "pt":
        return texto
    try:
        tradutor = GoogleTranslator(source="pt", target=idioma_destino)
        return tradutor.translate(texto)
    except Exception as e:
        return texto


def gerar_audio_resposta(texto, idioma):
    """Gera áudio da resposta usando gTTS."""
    mapa_gtts = {"pt": "pt", "en": "en", "es": "es", "fr": "fr",
                 "de": "de", "it": "it", "ja": "ja", "zh-CN": "zh-CN"}
    lang_tts = mapa_gtts.get(idioma, "pt")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
        tts = gTTS(text=texto, lang=lang_tts)
        tts.save(tmp_audio.name)
        tmp_audio_path = tmp_audio.name

    with open(tmp_audio_path, "rb") as f:
        audio_bytes = f.read()
    os.unlink(tmp_audio_path)
    return audio_bytes


def gerar_resposta_llm(pergunta, contexto, api_key):
    """Gera resposta usando a Groq API com contexto RAG."""
    prompt = f"""Você é o concierge virtual do Hotel LuxStay, um hotel 5 estrelas em Florianópolis.
Seja educado, profissional e acolhedor. Use SOMENTE as informações abaixo para responder.
Se a pergunta não puder ser respondida com o contexto, diga educadamente que não tem essa 
informação e sugira contato com a recepção pelo (48) 3333-1500.

Contexto do Hotel:
{contexto}

Pergunta do hóspede:
{pergunta}

Responda de forma clara, objetiva e hospitaleira."""

    client = Groq(api_key=api_key)
    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content


# ---------- Inicializar histórico de chat ----------
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []
    # Mensagem de boas-vindas
    st.session_state.mensagens.append({
        "role": "bot",
        "content": "Olá! Sou o concierge virtual do Hotel LuxStay. 🌟\n\nEstou aqui para ajudá-lo com informações sobre quartos, serviços, restaurantes, passeios e muito mais. Como posso tornar sua estadia perfeita?",
        "audio": None,
    })

# ---------- Renderizar histórico de mensagens ----------
for msg in st.session_state.mensagens:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="msg-user">
            <div class="bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="msg-bot">
            <div class="avatar">🏨</div>
            <div class="bubble">{msg["content"]}</div>
        </div>
        """, unsafe_allow_html=True)
        if msg.get("audio"):
            st.audio(msg["audio"], format="audio/mp3")

# ---------- Área de input ----------
st.markdown("---")

# Modo de entrada
modo = st.radio(
    "Modo de entrada:",
    ["⌨️ Texto", "🎙️ Áudio"],
    horizontal=True,
    label_visibility="collapsed",
)

texto_usuario = ""

if modo == "🎙️ Áudio":
    audio_file = st.file_uploader(
        "Envie um áudio (mp3, wav, m4a, ogg):",
        type=["mp3", "wav", "m4a", "ogg"],
        label_visibility="collapsed",
    )

    if audio_file:
        st.audio(audio_file)
        if st.button("📝 Transcrever", use_container_width=True):
            with st.spinner("🎙️ Transcrevendo..."):
                try:
                    transcricao = transcrever_audio(audio_file)
                    st.session_state["transcricao"] = transcricao
                except Exception as e:
                    st.error(f"Erro: {e}")

        if "transcricao" in st.session_state:
            st.success(f"📝 \"{st.session_state['transcricao']}\"")
            texto_usuario = st.session_state["transcricao"]
else:
    texto_usuario = st.text_area(
        "Mensagem:",
        height=80,
        placeholder="Digite sua pergunta aqui...",
        label_visibility="collapsed",
    )

# ---------- Sugestões rápidas ----------
st.markdown("")
cols = st.columns(4)
sugestoes = [
    ("🕐 Check-in", "Qual o horário de check-in e check-out?"),
    ("🍳 Café", "O café da manhã está incluso? Qual horário?"),
    ("🏊 Piscina", "O hotel tem piscina? Qual o horário?"),
    ("💰 Quartos", "Quais tipos de quarto estão disponíveis e os preços?"),
]

for i, (label, pergunta) in enumerate(sugestoes):
    with cols[i]:
        if st.button(label, use_container_width=True, key=f"sug_{i}"):
            texto_usuario = pergunta

# ---------- Botão enviar ----------
col_enviar, col_limpar = st.columns([4, 1])

with col_enviar:
    enviar = st.button("➤ Enviar", use_container_width=True, type="primary", disabled=not texto_usuario)

with col_limpar:
    if st.button("🗑️", use_container_width=True, help="Limpar conversa"):
        st.session_state.mensagens = [{
            "role": "bot",
            "content": "Conversa limpa! Como posso ajudá-lo?",
            "audio": None,
        }]
        st.rerun()

# ---------- Processar pergunta ----------
if enviar and texto_usuario:
    if not api_key:
        st.error("🔑 Configure sua Groq API Key na barra lateral.")
    elif not rag.esta_carregado():
        st.error("📚 Carregue a base de conhecimento na barra lateral primeiro.")
    else:
        # Adicionar mensagem do usuário
        st.session_state.mensagens.append({
            "role": "user",
            "content": texto_usuario,
            "audio": None,
        })

        with st.spinner(""):
            # 1. RAG
            contexto = rag.buscar_contexto(texto_usuario)

            # 2. LLM
            resposta_pt = gerar_resposta_llm(texto_usuario, contexto, api_key)

            # 3. Tradução
            if codigo_idioma != "pt":
                resposta_final = traduzir_texto(resposta_pt, codigo_idioma)
            else:
                resposta_final = resposta_pt

            # 4. Áudio
            try:
                audio_bytes = gerar_audio_resposta(resposta_final, codigo_idioma)
            except:
                audio_bytes = None

        # Adicionar resposta do bot
        st.session_state.mensagens.append({
            "role": "bot",
            "content": resposta_final,
            "audio": audio_bytes,
        })

        # Limpar transcrição se existir
        if "transcricao" in st.session_state:
            del st.session_state["transcricao"]

        st.rerun()

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    '<p style="text-align:center; color:#64748b; font-size:0.75rem;">'
    '🏨 Hotel LuxStay • Av. Beira Mar, 1500 • Florianópolis/SC<br>'
    'Powered by Groq + Whisper + RAG + gTTS</p>',
    unsafe_allow_html=True,
)
