import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"  # You can replace this with a larger model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a HuggingFacePipeline
pipeline = HuggingFacePipeline(pipeline=model)

# Create a vector store
embeddings = HuggingFaceEmbeddings()
vector_store = FAISS.from_texts(["Your knowledge base text here"], embeddings)

# Create a text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# Define a prompt template
template = """
Context: {context}

