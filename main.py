#!/usr/bin/env python
from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langserve import add_routes

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

"""
NVIDIA annual report 2024 : https://s201.q4cdn.com/141608511/files/doc_financials/2024/ar/NVIDIA-2024-Annual-Report.pdf
"""

# Read data from text file to gain fortune knowledge
raw_documents = PyPDFLoader("data/NVIDIA-2024-Annual-Report.pdf").load_and_split()

# Chunk with RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5120, chunk_overlap=200)
documents = text_splitter.split_documents(raw_documents)

# Embedding text into vector and store into vector database
embedding_model = OpenAIEmbeddings()

db = FAISS.from_documents(documents=documents, 
                           embedding=embedding_model)

# Set type of retriever
retriever = db.as_retriever()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

template = """Answer the question based on the following context only:

{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
model = ChatOpenAI()


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# 4. App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# 5. Adding chain route
add_routes(
    app,
    chain,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)