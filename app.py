from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from chainlit.types import ThreadDict
import chainlit as cl
import dotenv


dotenv.load_dotenv()
    
model = ChatOpenAI()

def create_chain(docs, model):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5120, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # Embedding text into vector and store into vector database
    embedding_model = OpenAIEmbeddings()

    db = FAISS.from_documents(documents=documents, 
                            embedding=embedding_model)

    # Set type of retriever
    retriever = db.as_retriever()
    
    template = """Answer the question based on the following context only:

    {context}

    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)


    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])


    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
        
    return chain


@cl.set_chat_profiles
async def set_profile():
    return [
        cl.ChatProfile(
            name="Assistant",
            markdown_description="I am your helpful assistant, ready to answer your questions.",
            icon="/public/avatar.png"  # Replace with the actual path to your image
        )
    ]
    
    
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    mock_user = {
        "user_1" : ("user1", "pass"),
        "user_2" : ("user2", "pass")
    }
    # 
    if (username, password) in mock_user.values():
        return cl.User(
            identifier=username, metadata={"role": "user", "provider": "credentials"}
        )
    else:
        return None
    

@cl.on_chat_start
async def on_chat_start():    
    file = await cl.AskFileMessage(
        content="Please upload a python file to begin!", accept=["application/pdf"]
      ).send()
    
    docs = PyPDFLoader(file[0].path).load_and_split()
    chain = create_chain(docs, model)
    
    cl.user_session.set("runnable", chain)
    

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
    
    
@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("runnable")
    res = rag_chain.invoke(message.content)
    await cl.Message(content=res).send()
