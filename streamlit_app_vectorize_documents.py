import streamlit as st
from sqlalchemy import text
from pathlib import Path
from psycopg.types.json import Jsonb
from pgvector.utils import to_db
import asyncio

from helpers import embed_directory, count_directory_tokens

st.set_page_config(
    page_title="Streamlit Documentation Vectorizer App",
    page_icon="🕹",
    initial_sidebar_state="collapsed",
)

st.header("Streamlit Documentation Vectorizer App 🎈🤖")
init_db_script = Path("pg/init_db.sql").read_text()

conn = st.experimental_connection("database", type="sql")

def echo_query(q: str):
    st.subheader("query:")
    st.write(
        f"""\
```sql
{q}          
```"""
    )
    st.subheader("result:")
    result = conn.query(q)
    result

if st.button("Initialize DB"):
    with conn.session as s:
        s.execute(text(init_db_script))
        s.commit()

with st.expander("Show initialization code"):
    st.code(init_db_script)



with st.expander("Show Documents DB Data"):
    q = """\
    SELECT *
    FROM documents;
    """
    st.write(
        f"""\
```sql
{q}          
```"""
    )
    st.subheader("result:")
    result = conn.query(q)
    result

with st.form("directory"):
    documents_directory = st.text_input("Documents Directory", "docs/content/library/get-started")
    is_submitted = st.form_submit_button()
num_tokens = count_directory_tokens(documents_directory)
st.warning(f"This directory will cost ~`{num_tokens}` tokens or ${(num_tokens * (0.0001 / 1000)):.4f}")
if not is_submitted:
    st.stop()

docs_batches = asyncio.run(embed_directory(documents_directory))
with conn.session as s:
    with s.connection().connection.cursor() as cursor:
        for docs in docs_batches:
            with cursor.copy(
                "COPY documents (content, metadata, embedding) FROM STDIN;"
            ) as copy:
                for doc in docs:
                    st.write(doc.metadata)
                    copy.write_row(
                        (
                            doc.content,
                            Jsonb(doc.metadata),
                            to_db(doc.embedding, 1536),
                        )
                    )
    s.execute(text("commit;"))

q = """\
    SELECT *
    FROM documents;
    """

st.write(
        f"""\
```sql
{q}          
```"""
    )
st.subheader("result:")
result = conn.query(q, ttl=1)
result