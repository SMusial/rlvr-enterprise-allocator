import streamlit as st

st.set_page_config(
    page_title="RLVR Enterprise Allocator",
    page_icon="🤖",
    layout="wide",
)

# --- language selector at very top of sidebar ---
lang = st.sidebar.radio(
    "🌐 Language / Langue / Idioma / Język",
    options=["EN", "FR", "ES", "PL"],
    horizontal=True,
)
st.session_state["lang"] = lang

# --- chapter selector ---
chapter = st.sidebar.selectbox(
    "📚 Chapter" if lang == "EN" else
    "📚 Chapitre" if lang == "FR" else
    "📚 Capítulo" if lang == "ES" else
    "📚 Rozdział",
    options=[f"Chapter {i:02d}" for i in range(1, 21)],
)

ch_num = int(chapter.split()[-1])

if ch_num == 1:
    from chapters.ch01 import render
    render()
elif ch_num == 2:
    from chapters.ch02 import render
    render()
elif ch_num == 3:
    from chapters.ch03 import render
    render()
else:
    st.info(
        f"🚧 Chapter {ch_num:02d} is not yet implemented." if lang == "EN" else
        f"🚧 Chapitre {ch_num:02d} n'est pas encore implémenté." if lang == "FR" else
        f"🚧 Capítulo {ch_num:02d} aún no está implementado." if lang == "ES" else
        f"🚧 Rozdział {ch_num:02d} nie jest jeszcze zaimplementowany."
    )
