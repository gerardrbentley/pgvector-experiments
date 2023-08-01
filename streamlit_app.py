import streamlit as st

from helpers import cached_embed
from pgvector.utils import to_db

q = """\
select
    content,
    metadata,
    similarity
from match_documents(:question_embedding, 5);
"""

st.set_page_config(
    page_title="Streamlit Documentation Search App",
    page_icon="🕹",
    initial_sidebar_state="collapsed",
)
st.header("Streamlit Documentation Search App 🎈🤖")

conn = st.experimental_connection("database", type="sql", pool_pre_ping=True)

with st.form("Question"):
    question = st.text_area(
        "Enter your Streamlit Question:", "How do I connect to a database?"
    )
    submitted = st.form_submit_button()

if not submitted:
    st.info("Submit a question to continue")
    st.stop()

st.toast("Getting Embedding", icon="🧠")
question_embedding = cached_embed(question)

st.toast("Getting Related Documentation", icon="📚")
similar_docs = conn.query(
    q, params={"question_embedding": to_db(question_embedding.embedding, 1536)}
)

st.header("Related Documentation:")
for doc in similar_docs.itertuples():
    st.write(doc.metadata)
    st.write(f"Similarity: `{doc.similarity}`")
    with st.expander("Show Doc"):
        st.markdown(doc.content)
