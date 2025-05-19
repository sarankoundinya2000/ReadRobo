import os
import shutil
import json
import PIL.Image
import pdf2image
import google.generativeai as genai
import re
import chromadb
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv('api.env')

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
load_dotenv('groqapi.env')

client = Groq(
    api_key=os.environ['GROQ_API_KEY'],
)

st.markdown(
    """
    <style>
    .reportview-container {
        background: linear-gradient(to right, #141e30, #243b55);
        color: white;
    }
    .st-chat-message {
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        max-width: 80%;
    }
    .st-user {
        background-color: #1E90FF;
        color: white;
        align-self: flex-end;
    }
    .st-bot {
        background-color: #696969;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Constants
OUTPUT_DIR_NAME = 'Indexed_Documents'
CHROMA_COLLECTION_NAME = 'documents'

# Streamlit styling

class EmbeddingFunction:
    pass

class Documents:
    pass

class Embeddings:
    pass

class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

def create_chroma_db(documents: list[str], path: str, name: str):
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_or_create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
    for i, d in enumerate(documents):
        db.add(documents=[d], ids=[str(i)])
    return db, name

def convert_pdf_to_images(pdf_path):
    images = pdf2image.convert_from_path(pdf_path)
    return images

def is_pdf(file_path):
    return file_path.lower().endswith('.pdf')

def get_relevant_passage(query: str, db, n_results: int):
    results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

def make_rag_prompt(query: str, relevant_passage: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below...
    QUESTION: '{query}'
    PASSAGE: '{escaped_passage}'
    ANSWER:
    """
    return prompt
    
def generate_answer(prompt: str):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are the best bot to answer the questions"},
            {"role": "user", "content": prompt + " give response in one line"}
        ],
        temperature=0.2,
        max_tokens=150,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content
    
def process_query_and_generate_answer(query, relevant_passage):
    prompt = make_rag_prompt(query, relevant_passage[0])
    answer = generate_answer(prompt)
    return answer
    
def extract_text_from_img(img):
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(["Extract information from the images - like each and every detail from the image ", img])
    response.resolve()
    
    if not response.parts:
        raise ValueError("Failed to extract text from image")
    
    return response.text

def extract_text_from_file(file_path):
    if is_pdf(file_path):
        images = convert_pdf_to_images(file_path)
        extracted_texts = [extract_text_from_img(image) for image in images]
        return '\n'.join(extracted_texts)
    else:
        img = PIL.Image.open(file_path)
        return extract_text_from_img(img)

def determine_document_type_with_llm(extracted_text):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": """You have to classify the given document based on the text: {birth certificate registry}
            For this text, you can give a label as 'Birth Certificate'. Only give the name in 1 to 2 words."""},
            {"role": "user", "content": f"Give response in one or two words: Classify the type of document based on the following text:\n {extracted_text} g"}
        ],
        temperature=0.5,
        max_tokens=150,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    doc_type = completion.choices[0].message.content.strip().lower()
    doc_type = re.sub(r'[^a-zA-Z0-9\s]', '', doc_type)
    doc_type = doc_type[:20]
    print(f"Generated document type: {doc_type}")
    return doc_type

def process_documents_in_folder(folder_path):
    index = {}
    document_texts = {} 
    
    if not os.path.exists(OUTPUT_DIR_NAME):
        os.makedirs(OUTPUT_DIR_NAME)

    extracted_text_all = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        print(file_path)
        if not (file_name.lower().endswith('.pdf') or file_name.lower().endswith(('.png', '.jpg', '.jpeg'))):
            continue
        
        try:
            extracted_text = extract_text_from_file(file_path)
            extracted_text = extracted_text.lower()
            extracted_text = re.sub(r'[^\w\s]', '', extracted_text)
            extracted_text = ' '.join(extracted_text.split())
            doc_type = determine_document_type_with_llm(extracted_text)
            extracted_text_all.append(doc_type + ' ' + extracted_text)
            
        except ValueError as e:
            print(f"Error processing file {file_name}: {e}")
            continue
     
    db, db_name = create_chroma_db(extracted_text_all, './ocr', 'OCR_database3')
    return db, db_name
    
def show_chroma_db_contents(db,up_files):
    # Fetch all documents from the database, along with their metadata
    results = db.get(include=["documents"])

    st.write("### Documents stored in ChromaDB:")

    # Display each document and its corresponding metadata
    i=1
    for doc,file in zip(results['documents'][:-1],up_files):
        
        st.write(f"**Document:** {i}")
        st.write(f"**Document content:** {doc}")
        st.write("-" * 80)
        i+=1
# Streamlit app title
st.title('🧠 Smart Document Q&A Chatbot')



# Sidebar
st.sidebar.title("App info & Settings")
st.sidebar.header("About")
st.sidebar.info(""" 
## 🧠 Smart Document Q&A Chatbot:
This application allows you to upload documents and ask questions based on the text extracted from those documents. You can ask multiple questions, and the responses will be displayed interactively.""")

st.sidebar.header("Select to View Extracted Text")
# Sidebar option to show extracted text
show_text_option = st.sidebar.checkbox("Show Extracted Text")

# Responses history
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

# Main page: File Upload
st.subheader("Upload files")
uploaded_files = st.file_uploader("Upload files (img, pdf, jpeg)", type=['jpg', 'jpeg', 'png', 'pdf'], accept_multiple_files=True)

# Save uploaded files
upload_folder_path = 'uploaded_files'
if not os.path.exists(upload_folder_path):
    os.makedirs(upload_folder_path)

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = os.path.join(upload_folder_path, uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success(f'Successfully uploaded {uploaded_file.name}')

# Main page: Query input
ui_prompt = st.chat_input("Enter your question")

db, db_name = process_documents_in_folder(upload_folder_path)

if ui_prompt:

   
    
    relevant_passage = get_relevant_passage(ui_prompt, db, 3)
    answer = process_query_and_generate_answer(ui_prompt, relevant_passage)
    answer = answer.replace("According to the passage, ", "")
    answer = answer.capitalize()

    # Append user question and bot answer to session state
    st.session_state['responses'].append(("user", ui_prompt))
    st.session_state['responses'].append(("bot", answer))

# Process documents and show ChromaDB contents if checkbox is selected
if show_text_option and uploaded_files:
    show_chroma_db_contents(db,uploaded_files)

# Display chat history
for role, message in st.session_state['responses']:
    if role == 'user':
        st.write(f"**You:** {message}")
    else:
        st.write(f"**Bot:** {message}")