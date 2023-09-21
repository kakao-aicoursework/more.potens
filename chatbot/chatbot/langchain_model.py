import os
from pprint import pprint

from langchain import LLMChain
from langchain.chains import SequentialChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma

# openai.api_key = "<YOUR_OPENAI_API_KEY>"
os.environ["OPENAI_API_KEY"] = open("../../apikey.txt", "r").read()

CUR_DIR = os.path.dirname(os.path.abspath('.'))
PROJECT_DATA_KAKAOSYNC = os.path.join(CUR_DIR, "datas/project_data_kakaosync.txt")
PROMPT_TEMPLATE = os.path.join(CUR_DIR, "templates/template_with_embedding.txt")

DATA_DIR = os.path.dirname(os.path.abspath('../datas'))
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "chroma-persist")
CHROMA_COLLECTION_NAME = "dosu-bot"


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path),
        ),
        output_key=output_key,
        verbose=True,
    )


def generate_answer(question) -> dict[str, str]:
    writer_llm = ChatOpenAI(temperature=0.1, max_tokens=1024, model="gpt-3.5-turbo-16k")

    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 질문 체인 생성
    question_chain = ConversationalRetrievalChain.from_llm(writer_llm,
                                                           retriever=db.as_retriever(),
                                                           memory=memory,
                                                           verbose=True)

    preprocess_chain = SequentialChain(
        chains=[
            question_chain
        ],
        input_variables=["data", "question"],
        output_variables=["answer"],
        verbose=True,
    )

    context = dict(
        data=read_prompt_template(PROJECT_DATA_KAKAOSYNC),
        question=question,
    )

    context = question_chain.invoke(question)

    return context


if __name__ == "__main__":
    q = "서비스에 카카오싱크를 도입하는 방법을 알려주세요."
    pprint(generate_answer(q))
