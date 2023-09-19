import os
from pprint import pprint

from langchain import LLMChain
from langchain.chains import SequentialChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


CUR_DIR = os.path.dirname(os.path.abspath('chatbot'))
PROJECT_DATA_KAKAOSYNC = os.path.join(CUR_DIR, "datas/project_data_kakaosync.txt")
PROMPT_TEMPLATE = os.path.join(CUR_DIR, "datas/template_first.txt")


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

    # 질문 체인 생성
    question_chain = create_chain(writer_llm, PROMPT_TEMPLATE, "answer")

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
    context = question_chain(context)

    return context["answer"]


if __name__ == "__main__":
    question = "서비스에 카카오싱크를 도입하는 방법을 알려주세요."
    pprint(generate_answer(question))
