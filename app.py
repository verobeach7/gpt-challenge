from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.base import BaseCallbackHandler
from operator import itemgetter
import streamlit as st
import os


st.set_page_config(
    page_title="Streamlit is 🔥",
    page_icon="🔥",
)


class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self, *args, **kwargs):
        self.message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    directory = os.path.dirname(file_path)

    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"Directory created: {directory}")

    try:
        with open(file_path, "wb") as f:
            f.write(file_content)
    except Exception as e:
        print(f"An error occurred: {e}")

    cache_basic_path = f"./.cache/embeddings/{file.name}"
    cache_directory = os.path.dirname(cache_basic_path)
    cache_path = LocalFileStore(cache_basic_path)

    if not os.path.exists(cache_directory):
        os.makedirs(cache_directory, exist_ok=True)
        print(f"Cache Directory created: {cache_directory}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_path)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


def load_memory(_):
    return st.session_state["memory"].load_memory_variables({})["history"]


def get_history():
    return st.session_state["memory"].load_memory_variables({})


def invoke_chain(question):
    result = chain.invoke(question)
    st.session_state["memory"].save_context(
        {"input": question},
        {"output": result.content},
    )


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Answer the question using ONLY the following context. If you don't know the answer just say you don't know. DON'T make anything up.

            Context: {context}
            """,
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


st.title("Streamlit DOC-GPT")

st.markdown(
    """
Welcome!

Use this chatbot to ask questions to an AI about your file!

### Step 1. Add your OpenAI API Key

### Step 2. Upload your file

### Step 3. Aks questions to an AI
"""
)


with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI_API_KEY", placeholder="Add your OpenAI API Key", type="password"
    )
    file = None
    if openai_api_key:
        file = st.file_uploader(
            "Upload a .txt .pdf or .docx file",
            type=["pdf", "txt", "docx"],
        )

if openai_api_key:
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        temperature=0.1,
        streaming=True,
        callbacks=[ChatCallbackHandler()],
    )

    if file:
        retriever = embed_file(file)
        send_message("I'm ready! Ask away!", "ai", save=False)
        paint_history()
        message = st.chat_input("Ask anything about your file...")
        if message:
            send_message(message, "human")
            chain = (
                {
                    "context": retriever | RunnableLambda(format_docs),
                    "question": RunnablePassthrough(),
                }
                | RunnablePassthrough.assign(
                    history=RunnableLambda(
                        st.session_state["memory"].load_memory_variables
                    )
                    | itemgetter("history")
                )
                | prompt
                | llm
            )
            with st.chat_message("ai"):
                invoke_chain(message)
                print(get_history())
    else:
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "memory" not in st.session_state:
            st.session_state["memory"] = ConversationBufferMemory(
                llm=llm,
                max_token_limit=2000,
                return_messages=True,
            )
