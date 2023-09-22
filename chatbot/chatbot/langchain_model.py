import os
from pprint import pprint

from langchain import LLMChain, ConversationChain
from langchain.chains import SequentialChain, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma

CUR_DIR = os.path.dirname(os.getcwd()) + '/chatbot'
PROJECT_DATA_KAKAOSYNC = os.path.join(CUR_DIR, "datas/project_data_kakaosync.txt")
PROMPT_TEMPLATE = os.path.join(CUR_DIR, "templates/template_with_embedding.txt")

INTENT_PROMPT_TEMPLATE = os.path.join(CUR_DIR, "templates/template_intent.txt")
INTENT_LIST = os.path.join(CUR_DIR, "templates/intent_list.txt")

KAKAO_SYNC_PROMPT = os.path.join(CUR_DIR, "templates/template_kakaosync.txt")
KAKAO_SOCIAL_PROMPT = os.path.join(CUR_DIR, "templates/template_kakaosocial.txt")
TALK_CHANNEL_PROMPT = os.path.join(CUR_DIR, "templates/template_talkchannel.txt")

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "chroma-persist")
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
    llm = ChatOpenAI(temperature=0.1, max_tokens=1024, model="gpt-3.5-turbo-16k")

    db = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=OpenAIEmbeddings(),
        collection_name=CHROMA_COLLECTION_NAME,
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # 의도 체인 생성
    parse_intent_chain = create_chain(
        llm=llm,
        template_path=INTENT_PROMPT_TEMPLATE,
        output_key="intent",
    )

    context = dict(
        input=question,
        intent_list=read_prompt_template(INTENT_LIST),
        question=question,
    )

    intent = parse_intent_chain.run(context)

    kakao_sync_chain = create_chain(
        llm=llm,
        template_path=KAKAO_SYNC_PROMPT,
        output_key="output",
    )

    kakao_social_chain = create_chain(
        llm=llm,
        template_path=KAKAO_SOCIAL_PROMPT,
        output_key="output",
    )

    talk_channel_chain = create_chain(
        llm=llm,
        template_path=TALK_CHANNEL_PROMPT,
        output_key="output",
    )

    default_chain = ConversationChain(llm=llm, output_key="output")

    if intent == "kakaosync":
        context["context"] = db.similarity_search(context["question"])
        result = kakao_sync_chain.run(context)
    elif intent == "kakkosocial":
        context["context"] = db.similarity_search(context["question"])
        result = kakao_social_chain.run(context)
    elif intent == "talkchannel":
        context["context"] = db.similarity_search(context["question"])
        result = talk_channel_chain.run(context)
    else:
        result = default_chain.run(context["input"])

    return result


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = open("../../apikey.txt", "r").read()

    q = "카카오싱크를 서비스에 도입하는 방법은 무엇인가요?"
    r = generate_answer(q)
    pprint(r)
