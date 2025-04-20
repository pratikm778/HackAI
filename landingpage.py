import streamlit as st

main = st.Page(
    "./landing.py",
    title="Home Page",
)

dca = st.Page(
    "./runner.py",
    title="Multi-Modal RAG Agent",
)

swp = st.Page(
    "./2ndrunner.py",
    title="About LTIMindtree",
)


pg = st.navigation(pages=[main, dca, swp])
pg.run()
