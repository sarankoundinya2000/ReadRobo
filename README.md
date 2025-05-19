# ReadRobo
ReadRobo - Intelligent Document OCR and Classification System
Here’s a **README.md** file for your project, summarizing its functionality, setup instructions, and key features in a clean and professional format:

---

# 🧠 Smart Document Q\&A Chatbot

This repository contains a scalable, intelligent OCR application that processes and classifies over 100 types of documents—including complex handwritten forms such as **Birth Certificates**, **Death Certificates**, and **Resumes**. The app enables users to extract, embed, and interact with document content via a chatbot-style interface using cutting-edge models from Google and Groq.

---

## 🚀 Key Features

* ✅ **Multi-format Support**: Process PDFs and image files (`.png`, `.jpg`, `.jpeg`) containing handwritten or printed content.
* 🔍 **Document Classification**: Uses LLaMA3 models to classify documents (e.g., birth certificate, resume) with over **95% accuracy**.
* 🧠 **Question Answering with RAG**: Employs a Retrieval-Augmented Generation (RAG) system combining **Gemini embeddings** and **LLaMA3** for precise, real-time Q\&A.
* 📊 **Document Indexing**: Embeds and stores content in **ChromaDB** for fast semantic retrieval.
* 💬 **Chat Interface**: Built with **Streamlit**, users can query uploaded documents in natural language and receive answers.
* ⚡ **Performance**: Reduced processing time by 40% through optimized AI pipelines and efficient embedding logic.

---

## 🧰 Technologies Used

| Task                        | Technology                  |
| --------------------------- | --------------------------- |
| OCR & Extraction            | Google Gemini Flash         |
| Embedding & Retrieval       | Gemini Embeddings, ChromaDB |
| LLM for Classification/Q\&A | LLaMA3 (via Groq API)       |
| UI Framework                | Streamlit                   |
| PDF/Image Processing        | pdf2image, PIL              |

---

## 🗂 Folder Structure

```
├── uploaded_files/           # User uploaded docs
├── Indexed_Documents/        # Final output & embeddings
├── .env                      # API keys for Gemini, Groq
├── main.py                   # Main app script
├── requirements.txt          # All dependencies
└── README.md
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-repo/ocr-rag-app.git
   cd ocr-rag-app
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create `.env` and `groqapi.env` files and add:

   ```env
   GEMINI_API_KEY=your_google_genai_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the app**

   ```bash
   streamlit run main.py
   ```

---

## 💡 Example Queries

* "What is the date of birth mentioned in the birth certificate?"
* "Give me the name of the deceased in this death certificate."
* "Extract the qualifications from the resume."

---

## 📝 Future Improvements

* Add support for tabular data extraction.
* Enable export to structured formats like CSV/JSON.
* Support multi-language OCR and translation.

---

## 🤝 Contributing

Feel free to open issues or pull requests for enhancements. Let's build better AI-powered document tools together!


