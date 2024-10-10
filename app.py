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
from langchain.schema.runnable.config import RunnableConfig
from langchain.chains import conversational_retrieval

dotenv.load_dotenv()
    
model = ChatOpenAI(model='gpt-4o-mini', streaming=True)

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
        
    cl.user_session.set("runnable", chain)


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
        content="Please upload a PDF file to begin!", accept=["application/pdf"]
      ).send()
    
    msg = cl.Message(content=f"Processing `{file[0].name}`...")
    await msg.send()
    
    docs = PyPDFLoader(file[0].path).load_and_split()
    create_chain(docs, model)

    

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    print("The user resumed a previous chat session!")
    
    
@cl.on_message
async def on_message(message: cl.Message):
    # memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")

    res = cl.Message(content="")

    async for chunk in runnable.astream(message.content):
        await res.stream_token(chunk)

    await res.send()

    # memory.chat_memory.add_user_message(message.content)
    # memory.chat_memory.add_ai_message(res.content)
