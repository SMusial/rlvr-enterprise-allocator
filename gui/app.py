import streamlit as st
import importlib
import sys

st.set_page_config(
    page_title="RLVR Enterprise Allocator",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 21.9rem !important; max-width: 21.9rem !important; }
</style>
""", unsafe_allow_html=True)

# Language
if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

lang = st.sidebar.radio(
    "🌐 Language / Sprache / Langue / Idioma / Język",
    options=["EN", "DE", "FR", "ES", "PL"],
    horizontal=True,
    index=["EN", "DE", "FR", "ES", "PL"].index(st.session_state["lang"]),
)
st.session_state["lang"] = lang

# Chapter
chapter_label = (
    "📚 Chapter"   if lang == "EN" else
    "📚 Kapitel"   if lang == "DE" else
    "📚 Chapitre"  if lang == "FR" else
    "📚 Capítulo"  if lang == "ES" else
    "📚 Rozdział"
)
chapter_options = [f"Chapter {i:02d}" for i in range(1, 21)]
if "chapter_index" not in st.session_state:
    st.session_state["chapter_index"] = 0

chapter = st.sidebar.selectbox(
    chapter_label,
    options=chapter_options,
    index=st.session_state["chapter_index"],
)
st.session_state["chapter_index"] = chapter_options.index(chapter)
ch_num = int(chapter.split()[-1])

# Routing — import ONCE, never delete from sys.modules
IMPLEMENTED = set(range(1, 14))
if ch_num in IMPLEMENTED:
    mod_name = f"chapters.ch{ch_num:02d}"
    if mod_name not in sys.modules:
        mod = importlib.import_module(mod_name)
    else:
        mod = sys.modules[mod_name]
    mod.render()
else:
    st.info(
        f"🚧 Chapter {ch_num:02d} is not yet implemented."                      if lang == "EN" else
        f"🚧 Kapitel {ch_num:02d} ist noch nicht implementiert."                if lang == "DE" else
        f"🚧 Chapitre {ch_num:02d} n'est pas encore implémenté."               if lang == "FR" else
        f"🚧 Capítulo {ch_num:02d} aún no está implementado."                  if lang == "ES" else
        f"🚧 Rozdział {ch_num:02d} nie jest jeszcze zaimplementowany."
    )
