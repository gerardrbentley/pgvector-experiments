# Example Streamlit Connection to PG Vector Database

Utilize:

- PG Vector Postgres database 
- OpenAI text embeddings 
- Streamlit
    - Featuring SQL connection

To create a semantic documentation search app

Live at [https://pgvector-test.streamlit.app/](https://pgvector-test.streamlit.app/)

## What's This?

Python Applications

- `streamlit_app.py`: Primary documentaiton search app
    - Embeds the user's question 
    - Performs a similarity search in the database to find relevant documentation pages
- `streamlit_app_vectorize_documents.py`: Admin App
    - Naive embedding and batched copy into Postgres database
    - (There's dozens of better ways to do this....)
        - Don't save document content in database: link the documents / S3 url
            - Postgres TOAST can help with this to a degree
        - Save the database records before attempting embedding
        - error handling
        - testing...
- `helpers.py`: Miscellanious embedding and chunking functions

Postgres Setup

- `pg/init_db.sql`: Basic table and function for utilizing PG Vector. Mostly from Langchain supabase demo
- `docker-compose.yaml`: Dockerized PG Vector database via `ankane/pgvector` image for running locally

## Local Setup

Assuming you have python and docker installed on your system:

```sh
# Spin up postgres
docker-compose up --build -d
# Initialize PG Vector db
docker-compose exec database psql -U demo_user demo -f /tmp/pg/init_db.sql
# Install python dependencies
python -m venv venv
. ./venv/bin/activate
python -m pip install -r requirements.txt
# Run streamlit admin (embed / upload docs directory app)
python -m streamlit run streamlit_app_vectorize_documents.py
# Run streamlit query app
python -m streamlit run streamlit_app.py
```

## Fly IO Setup

Running a Postgres cluster on fly.io is relatively straightforward (compared to running your own)

You can use a slightly customized image to add pgvector as well

```sh
fly pg create --image-ref gerardrbentley/ha-flypg-pgvector:latest
```

After initializing it and allocating an external IP address (as in Postgres [External Connection Docs](https://fly.io/docs/postgres/connecting/connecting-external/)), you should be able to access your cluster from anywhere.

You'll likely need to have an ipv4 address for connecting (ex. from Streamlit Community Cloud).

## Local Recreation

This example uses the streamlit documentation as a sample database

Those docs can be cloned from [https://github.com/streamlit/docs](https://github.com/streamlit/docs) and cost ~$0.02 to embed at the time of writing with OpenAI's `text-embedding-ada-002` model.
