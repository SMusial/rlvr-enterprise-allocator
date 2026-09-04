import streamlit as st

st.set_page_config(
    page_title="RLVR Enterprise Allocator",
    page_icon="🤖",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stSidebar"] { min-width: 20.8rem !important; max-width: 20.8rem !important; }
</style>
""", unsafe_allow_html=True)

# --- chapter selector ---
chapter = st.sidebar.selectbox(
    "📚 Chapter",
    options=[f"Chapter {i:02d}" for i in range(1, 21)],
)

ch_num = int(chapter.split()[-1])

if ch_num == 1:
    from chapters.ch01 import render; render()
elif ch_num == 2:
    from chapters.ch02 import render; render()
elif ch_num == 3:
    from chapters.ch03 import render; render()
elif ch_num == 4:
    from chapters.ch04 import render; render()
elif ch_num == 5:
    from chapters.ch05 import render; render()
elif ch_num == 6:
    from chapters.ch06 import render; render()
elif ch_num == 7:
    from chapters.ch07 import render; render()
elif ch_num == 8:
    from chapters.ch08 import render; render()
elif ch_num == 9:
    from chapters.ch09 import render; render()
elif ch_num == 10:
    from chapters.ch10 import render; render()
elif ch_num == 11:
    from chapters.ch11 import render; render()
elif ch_num == 12:
    from chapters.ch12 import render; render()
elif ch_num == 13:
    from chapters.ch13 import render; render()
elif ch_num == 14:
    from chapters.ch14 import render; render()
elif ch_num == 15:
    from chapters.ch15 import render; render()
elif ch_num == 16:
    from chapters.ch16 import render; render()
elif ch_num == 17:
    from chapters.ch17 import render; render()
else:
    st.info(f"🚧 Chapter {ch_num:02d} is not yet implemented.")
