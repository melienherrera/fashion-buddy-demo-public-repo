import streamlit as st
from PIL import Image as PILImage
import io, os, vertexai
from vertexai.preview.vision_models import MultiModalEmbeddingModel, Image
from astrapy.db import AstraDB, AstraDBCollection
from dotenv import find_dotenv, load_dotenv
import json

# Get environment variables
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

google_credentials = dict(st.secrets["google_credentials"])
# print(dict(google_credentials))

with open(".streamlit/secrets.json", "w") as outfile: 
    json.dump(google_credentials, outfile)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = ".streamlit/secrets.json"


# Use Gemini Pro Vision as our LLM + Embedding Model
vertexai.init(project=st.secrets["GCP_PROJECT_ID"])
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

# Connect to AstraDB    
astra_db = AstraDB(
    token = st.secrets["ASTRA_DB_TOKEN"],
    api_endpoint = st.secrets["ASTRA_API_ENDPOINT"])

# Connect to Collection
collection = AstraDBCollection(
    collection_name="shopping_buddy_demo", astra_db=astra_db
)

# Create category mapping from UI selections to backend DB
category_mapping = {
    "Tops": "TOPS",
    "Dresses/Jumpsuits": "DRESSES_JUMPSUITS",
    "Outerwear": "OUTERWEAR",
    "Bottoms": "BOTTOMS",
    "Accessories": "ACCESSORIES",
    "Activewear":"ACTIVEWEAR",
    "Shoes": "SHOES",
}

# Create embedding of user uploaded image using Gemini Pro Vision
def get_img_embeddings(img_reference, text=""):
    img = Image.load_from_file(img_reference)
    embeddings = model.get_embeddings(
      image=img,
      contextual_text=text
  )
    return embeddings.image_embedding

# Save the user's uploaded image into the Assests folder
def saveImage(image, file_name):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    img_byte_arr = img_byte_arr.getvalue()

    # Define the directory and file path
    directory = "./Assets"
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name)

    # Write the uploaded file to the Assets Folder
    with open(file_path, 'wb') as f:
        f.write(img_byte_arr)

    reference_url = os.path.join(directory, file_name)
    return reference_url

# Print the recommendations to the UI
def show_recommendations(documents, category=None):
  if category:
    st.write("Category:", category)
  for doc in documents:
    st.write(doc["product_name"], "$" + str(doc["price"]), ", Gender:", doc["gender"])
    st.write(doc["details"])
    st.write(doc["link"])
    st.write(("Similarity score: ", doc["$similarity"]))
    st.image(doc["product_images"], width=300)

#Use vector search to find the similar produts in the collection
def find_similar_products(uploaded_image_url, gender, clothing_types):    
    if clothing_types:
        for category in clothing_types:
            # Run multiple ANN searches with filtering to pull top matches
            # for each category requested
            search_prompt = """
                I am trying to find pieces of apparel that are similar to what is in this
                picture. Ignore the model and only focus on finding the most similar clothing.

                I only care about apparel that falls within the category contained
                within triple backticks:

            ```{category}```
            """.format(category=category)
            reference_embeddings = get_img_embeddings(
                uploaded_image_url,
                text=search_prompt
            )

            documents = collection.vector_find(
                reference_embeddings,
                limit=3,
                filter={"category": category, "gender": gender.lower()},
                include_similarity=True
            )
            show_recommendations(documents, category=category)
    else:
        # Run single ANN search across entire DB by gender
        search_prompt = """
        I am trying to find pieces of apparel that are similar to what is in this
        picture. Pretend as if there is no model in the image, only clothing.
        """
        reference_embeddings = get_img_embeddings(
            uploaded_image_url, 
            text=search_prompt
        )

        documents = collection.vector_find(
            reference_embeddings,
            limit=3,
            filter={"gender": gender.lower()},
        )

        show_recommendations(documents)


def main():
    # Display the Title + Description
    st.title("Fashion Buddy 🛍️")
    st.write("*Fashion Buddy is where Vector Search 🤝 Fashion.*") 
    st.write("Say goodbye to wardrobe dilemmas and hello to effortless style. It's designed to help YOU find the best outfits and clothing items! All you need to do is upload any outfit or piece of clothing, and we'll find outfits that are similar. Give it a try!")

    # Add DataStax Logo:
    with st.sidebar:
         st.image('streamlit_app/Assets/datastax-logo.svg')
         st.text('')

    # Add Clothing Filters header
    with st.sidebar:
        st.header("Clothing Filters")

        # Gender selection
        gender = st.radio ("Select a Gender:",('Men', 'Women'))

        # Clothing type selection
        st.write("👡 Select Types of Clothing 👕:")
        categories = category_mapping.keys()
        clothing_types = [cat for cat in categories if st.checkbox(cat)]
        converted_clothing_types = [category_mapping.get(cat, cat) for cat in clothing_types]

    # Button for user to download an example image
    with st.sidebar:
        st.write("⬇️ Download an example image:⬇️")
        st.download_button(
            label='Download Image',
            data=open('streamlit_app/Assets/example-outfit.jpeg', 'rb').read(),
            file_name='example-outfit.jpeg',
            mime='image/jpeg'
        )

    # Drag and drop file uploader
    uploaded_file = st.file_uploader("Upload any outfit or piece of clothing...", type=["jpg", "png", "jpeg"])

    # Display uploaded file
    if uploaded_file is not None:
        file_name = uploaded_file.name
        image = PILImage.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # Button to trigger embedding and search
        if st.button('Recommend Products'):
            with st.spinner('Finding similar products...'):
                reference_image_url = saveImage(image, file_name)
                find_similar_products(
                    reference_image_url, 
                    gender, 
                    converted_clothing_types
                )

if __name__ == "__main__":
    main()






