import importlib
import streamlit as st

st.set_page_config(page_title="RLVR Enterprise Allocator", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 20.8rem !important; max-width: 20.8rem !important; }
</style>
""", unsafe_allow_html=True)

st.sidebar.title("RLVR Enterprise Allocator")
st.sidebar.markdown("---")

def _on_chapter_change():
    opts = [f"Chapter {i:02d}" for i in range(1, 21)]
    st.session_state["chapter_index"] = opts.index(st.session_state["_chapter_select"])

opts = [f"Chapter {i:02d}" for i in range(1, 21)]
if "chapter_index" not in st.session_state:
    st.session_state["chapter_index"] = 0

st.sidebar.selectbox(
    "Chapter",
    opts,
    index=st.session_state["chapter_index"],
    key="_chapter_select",
    on_change=_on_chapter_change,
)

chapter = st.session_state["_chapter_select"]
mod_name = f"gui.chapters.ch{chapter.split()[1]}"

try:
    mod = importlib.import_module(mod_name)
    importlib.reload(mod)
    mod.render()
except ModuleNotFoundError:
    st.warning(f"Chapter module `{mod_name}` not found.")
except Exception as e:
    st.exception(e)
