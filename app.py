import os
import tempfile
import streamlit as st
from groq import Groq
from gtts import gTTS
import whisper
from dotenv import load_dotenv
import rag

load_dotenv()

st.set_page_config(page_title="MoveBus", page_icon="🚌", layout="centered")

st.title("🚌 MoveBus")
st.caption("Assistente de transporte público de Joinville")

# ---------- Sidebar: configurações ----------
with st.sidebar:
    st.header("Configurações")

    api_key = st.text_input(
        "Groq API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Obtenha em https://console.groq.com",
    )

    st.divider()

    if st.button("🔄 Carregar linhas de ônibus", use_container_width=True):
        log_area = st.empty()
        logs = []

        def log(msg):
            logs.append(msg)
            log_area.text("\n".join(logs))

        with st.spinner("Carregando dados..."):
            try:
                rag.carregar_linhas(status_callback=log)
                st.success("Base de dados pronta!")
            except Exception as e:
                st.error(str(e))

    if rag.esta_carregado():
        st.success("Base carregada ✅")
    else:
        st.warning("Base não carregada")

# ---------- Área principal ----------
st.subheader("Faça sua pergunta")

modo = st.radio("Modo de entrada", ["🎙️ Áudio", "⌨️ Texto"], horizontal=True)

texto_usuario = ""

if modo == "🎙️ Áudio":
    audio_file = st.file_uploader(
        "Envie um arquivo de áudio (mp3, wav, m4a, ogg)",
        type=["mp3", "wav", "m4a", "ogg"],
    )

    if audio_file:
        st.audio(audio_file)

        if st.button("📝 Transcrever áudio"):
            with st.spinner("Transcrevendo com Whisper..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(audio_file.name)[1]
                ) as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name

                model = whisper.load_model("base")
                resultado = model.transcribe(tmp_path)
                os.unlink(tmp_path)

                st.session_state["transcricao"] = resultado["text"]

        if "transcricao" in st.session_state:
            st.info(f"**Transcrição:** {st.session_state['transcricao']}")
            texto_usuario = st.session_state["transcricao"]

else:
    texto_usuario = st.text_area("Digite sua pergunta sobre as linhas de ônibus:", height=100)

# ---------- Gerar resposta ----------
if st.button("🚀 Perguntar", disabled=not texto_usuario, use_container_width=True):
    if not api_key:
        st.error("Informe a Groq API Key na barra lateral.")
    elif not rag.esta_carregado():
        st.error("Carregue as linhas de ônibus primeiro (barra lateral).")
    else:
        with st.spinner("Buscando resposta..."):
            contexto = rag.buscar_contexto(texto_usuario)

            prompt = f"""Você é um assistente de transporte público de Joinville.
Use SOMENTE as informações abaixo para responder.

Contexto:
{contexto}

Pergunta:
{texto_usuario}

Responda de forma simples, objetiva e amigável."""

            client = Groq(api_key=api_key)
            chat = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
            )
            resposta = chat.choices[0].message.content

        st.subheader("Resposta")
        st.write(resposta)

        # Gerar áudio da resposta
        with st.spinner("Gerando áudio..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tts = gTTS(text=resposta, lang="pt")
                tts.save(tmp_audio.name)
                tmp_audio_path = tmp_audio.name

            with open(tmp_audio_path, "rb") as f:
                audio_bytes = f.read()
            os.unlink(tmp_audio_path)

        st.audio(audio_bytes, format="audio/mp3")
        st.download_button("⬇️ Baixar resposta em áudio", audio_bytes, "resposta.mp3", "audio/mp3")