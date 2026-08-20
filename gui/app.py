import streamlit as st

st.set_page_config(
    page_title="RLVR Enterprise Allocator",
    page_icon="🤖",
    layout="wide",
)

# Widen sidebar by ~12%
st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 23.5rem !important;
    max-width: 23.5rem !important;
}
</style>
""", unsafe_allow_html=True)

# Import all chapters ONCE at module level — avoids circular import on rerun
import sys, importlib

def _import_chapter(name):
    if name not in sys.modules:
        return importlib.import_module(f"chapters.{name}")
    return sys.modules[name]

# Language selector
if "lang" not in st.session_state:
    st.session_state["lang"] = "EN"

lang = st.sidebar.radio(
    "🌐 Language / Sprache / Langue / Idioma / Język",
    options=["EN", "DE", "FR", "ES", "PL"],
    horizontal=True,
    index=["EN", "DE", "FR", "ES", "PL"].index(st.session_state["lang"]),
    key="lang_radio",
)
st.session_state["lang"] = lang

# Chapter selector — preserved across language changes
chapter_label = (
    "📚 Chapter"        if lang == "EN" else
    "📚 Kapitel"        if lang == "DE" else
    "📚 Chapitre"       if lang == "FR" else
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
    key="chapter_select",
)
st.session_state["chapter_index"] = chapter_options.index(chapter)

ch_num = int(chapter.split()[-1])

# Route to chapter — use importlib to avoid re-import circular issues
CHAPTER_MAP = {i: f"ch{i:02d}" for i in range(1, 14)}

if ch_num in CHAPTER_MAP:
    mod = _import_chapter(CHAPTER_MAP[ch_num])
    mod.render()
else:
    st.info(
        f"🚧 Chapter {ch_num:02d} is not yet implemented."                      if lang == "EN" else
        f"🚧 Kapitel {ch_num:02d} ist noch nicht implementiert."                if lang == "DE" else
        f"🚧 Chapitre {ch_num:02d} n'est pas encore implémenté."    if lang == "FR" else
        f"🚧 Capítulo {ch_num:02d} aún no está implementado."   if lang == "ES" else
        f"🚧 Rozdział {ch_num:02d} nie jest jeszcze zaimplementowany."
    )
