import os
from pprint import pprint

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ["OPENAI_API_KEY"] = open("../../apikey.txt", "r").read()
CUR_DIR = os.path.dirname(os.path.abspath('.'))
DATA_DIR = os.path.dirname(os.path.abspath('../datas'))

CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


def upload_embedding_from_file(fp):
    documents = TextLoader(fp).load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    print(docs, end='\n\n\n')

    Chroma.from_documents(
        docs,
        OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


if __name__ == "__main__":
    for file in os.listdir(os.path.abspath('../datas')):
        file_path = os.path.join(DATA_DIR, 'datas', file)
        upload_embedding_from_file(file_path)
