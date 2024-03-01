# Fashion Buddy Demo - Astra Vector Search and Vertex AI

### Intro
Have you ever scrolled through photo sharing websites such as Instagram or Pinterest and had your eyes set on a specific outfit but did not know where to get the items? The Shopping Buddy lets you upload a photo of any outfit and returns the most similar pieces of apparel to recreate the outfit. 

### Overview
This repo contains a Streamlit app which uses Vertex AI, Gemini, RAG, and Astra DB to perform vector similarity search and multi-modal Retrieval Augmented Generation (RAG). The same code is available in a Colab here: [Link to Colab](https://colab.research.google.com/drive/1XQm_gBCZ-xRcUj4oaxcgNDqQtZLg2NpD#scrollTo=A2hnFk7YlWAN)
Try out Fashion Buddy [here!](https://fashion-buddy-demo.streamlit.app/)

### Before you get started
Make sure you have the following:
* A free Astra DB account. You can sign up with a free account [here](https://astra.datastax.com)
* A Vertex AI account. Sign up [here](https://cloud.google.com/vertex-ai?hl=en)

### Credentials
* Download your service account key as a json file from Google Cloud console. Record the path to this json file. [See instructions here](https://cloud.google.com/iam/docs/keys-create-delete)
* Create a `.env` file in the same directory as `fashion_buddy.py`
* Copy/paste the following into your `.env` file and replace with your environment variables.
```
GCP_PROJECT_ID = "<YOUR_GCP_PROJECT_ID>"
ASTRA_DB_TOKEN= "AstraCS:..."
ASTRA_API_ENDPOINT= "https://<DATABASE_ID>-<REGION>.apps.astra.datastax.com"
GOOGLE_APPLICATION_CREDENTIALS_PATH= <./PATH/TO/YOUR/GOOGLE_SERVICE_ACCOUNT_KEY.json>
```

### Run the Application
In your terminal, run `streamlit run fashion_buddy.py` from the same directory as `fashion_buddy.py`
