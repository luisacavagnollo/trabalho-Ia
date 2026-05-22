"""
Hotel LuxStay - Assistente Virtual Inteligente
Chatbot com RAG, Transcrição de Áudio, Tradução e Resposta por Voz.
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
)

# ---------- Estilos customizados ----------
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .stButton > button {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Header ----------
st.markdown("<div class='main-header'>", unsafe_allow_html=True)
st.title("🏨 Hotel LuxStay")
st.caption("Concierge Virtual - Atendimento Inteligente 24h")
st.markdown("</div>", unsafe_allow_html=True)

# ---------- Sidebar: configurações ----------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/hotel-building.png", width=80)
    st.header("⚙️ Configurações")

    api_key = st.text_input(
        "🔑 Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Obtenha em https://console.groq.com",
    )

    st.divider()

    # Seleção de idioma para tradução
    st.subheader("🌐 Idioma da Resposta")
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
    )
    codigo_idioma = idiomas[idioma_selecionado]

    st.divider()

    # Carregar base RAG
    if st.button("📚 Carregar Base de Conhecimento", use_container_width=True):
        log_area = st.empty()
        logs = []

        def log(msg):
            logs.append(msg)
            log_area.text("\n".join(logs))

        with st.spinner("Carregando dados do hotel..."):
            try:
                rag.carregar_base(status_callback=log)
                st.success("✅ Base de conhecimento pronta!")
            except Exception as e:
                st.error(f"Erro: {str(e)}")

    if rag.esta_carregado():
        st.success("📚 Base carregada ✅")
    else:
        st.warning("⚠️ Base não carregada")

    st.divider()
    st.markdown("**Recursos:**")
    st.markdown("- 🎙️ Transcrição de áudio (Whisper)")
    st.markdown("- 🤖 RAG com base do hotel")
    st.markdown("- 🌐 Tradução automática")
    st.markdown("- 🔊 Resposta em áudio (gTTS)")


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
        return texto  # Já está em português
    try:
        tradutor = GoogleTranslator(source="pt", target=idioma_destino)
        return tradutor.translate(texto)
    except Exception as e:
        st.warning(f"Erro na tradução: {e}. Mostrando em português.")
        return texto


def gerar_audio_resposta(texto, idioma):
    """Gera áudio da resposta usando gTTS."""
    # Mapear códigos para gTTS (alguns são diferentes)
    mapa_gtts = {
        "pt": "pt",
        "en": "en",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "it": "it",
        "ja": "ja",
        "zh-CN": "zh-CN",
    }
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
    prompt = f"""Você é o concierge virtual do Hotel LuxStay, um hotel de luxo em Florianópolis.
Seja educado, profissional e acolhedor. Use SOMENTE as informações abaixo para responder.
Se a pergunta não puder ser respondida com o contexto disponível, diga educadamente que 
não tem essa informação e sugira que o hóspede entre em contato com a recepção.

Contexto do Hotel:
{contexto}

Pergunta do hóspede:
{pergunta}

Responda de forma clara, objetiva e hospitaleira. Use emojis com moderação para tornar 
a conversa mais amigável."""

    client = Groq(api_key=api_key)
    chat = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile",
    )
    return chat.choices[0].message.content


# ---------- Área principal ----------
st.subheader("💬 Como posso ajudá-lo?")

# Seleção do modo de entrada
modo = st.radio(
    "Modo de entrada:",
    ["⌨️ Texto", "🎙️ Áudio"],
    horizontal=True,
    help="Escolha se deseja digitar ou enviar um áudio com sua pergunta.",
)

texto_usuario = ""

if modo == "🎙️ Áudio":
    st.info("🎙️ Envie um arquivo de áudio com sua pergunta. Suportamos MP3, WAV, M4A e OGG.")

    audio_file = st.file_uploader(
        "Selecione o arquivo de áudio:",
        type=["mp3", "wav", "m4a", "ogg"],
        help="O áudio será transcrito automaticamente usando Whisper.",
    )

    if audio_file:
        st.audio(audio_file)

        if st.button("📝 Transcrever Áudio", use_container_width=True):
            with st.spinner("🎙️ Transcrevendo com Whisper..."):
                try:
                    transcricao = transcrever_audio(audio_file)
                    st.session_state["transcricao"] = transcricao
                except Exception as e:
                    st.error(f"Erro na transcrição: {e}")

        if "transcricao" in st.session_state:
            st.success(f"**📝 Transcrição:** {st.session_state['transcricao']}")
            texto_usuario = st.session_state["transcricao"]

else:
    texto_usuario = st.text_area(
        "Digite sua pergunta:",
        height=100,
        placeholder="Ex: Qual o horário do café da manhã? / Tem piscina? / Como faço para pedir room service?",
    )

# ---------- Sugestões rápidas ----------
st.markdown("**💡 Perguntas frequentes:**")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🕐 Check-in/out", use_container_width=True):
        texto_usuario = "Qual o horário de check-in e check-out?"
with col2:
    if st.button("🍳 Café da manhã", use_container_width=True):
        texto_usuario = "O café da manhã está incluso? Qual o horário?"
with col3:
    if st.button("🏊 Piscina", use_container_width=True):
        texto_usuario = "O hotel tem piscina? Qual o horário?"

# ---------- Gerar resposta ----------
st.divider()

if st.button("🚀 Enviar Pergunta", disabled=not texto_usuario, use_container_width=True, type="primary"):
    if not api_key:
        st.error("🔑 Informe a Groq API Key na barra lateral.")
    elif not rag.esta_carregado():
        st.error("📚 Carregue a base de conhecimento primeiro (barra lateral).")
    else:
        # Mostrar a pergunta
        st.markdown(f"**🧑 Hóspede:** {texto_usuario}")

        with st.spinner("🤖 Buscando informações..."):
            # 1. Buscar contexto RAG
            contexto = rag.buscar_contexto(texto_usuario)

            # 2. Gerar resposta com LLM
            resposta_pt = gerar_resposta_llm(texto_usuario, contexto, api_key)

        # 3. Traduzir se necessário
        if codigo_idioma != "pt":
            with st.spinner(f"🌐 Traduzindo para {idioma_selecionado}..."):
                resposta_final = traduzir_texto(resposta_pt, codigo_idioma)
                st.info(f"🌐 Resposta traduzida para: **{idioma_selecionado}**")
        else:
            resposta_final = resposta_pt

        # 4. Exibir resposta
        st.markdown("---")
        st.markdown(f"**🤖 Concierge LuxStay:**")
        st.write(resposta_final)

        # 5. Gerar áudio da resposta
        with st.spinner("🔊 Gerando resposta em áudio..."):
            try:
                audio_bytes = gerar_audio_resposta(resposta_final, codigo_idioma)

                st.markdown("**🔊 Ouça a resposta:**")
                st.audio(audio_bytes, format="audio/mp3")
                st.download_button(
                    "⬇️ Baixar áudio da resposta",
                    audio_bytes,
                    "resposta_luxstay.mp3",
                    "audio/mp3",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Não foi possível gerar áudio: {e}")

        # Mostrar detalhes técnicos (expansível)
        with st.expander("🔍 Detalhes técnicos (RAG)"):
            st.markdown("**Contexto recuperado da base:**")
            st.text(contexto)
            if codigo_idioma != "pt":
                st.markdown("**Resposta original (PT):**")
                st.text(resposta_pt)

# ---------- Footer ----------
st.divider()
st.markdown(
    "<center><small>🏨 Hotel LuxStay - Concierge Virtual | "
    "Powered by Groq + Whisper + RAG + gTTS</small></center>",
    unsafe_allow_html=True,
)
