from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import DASHSCOPE_API_KEY, BASE_URL, LLM_MODEL


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


SYSTEM_PROMPT = (
    "你是一个基于文档内容进行问答的助手。"
    "请使用以下检索到的文档内容来回答用户的问题。"
    "如果你无法从文档中找到答案，请诚实地说明你不知道，"
    "不要编造信息。请尽量用八句话以内简洁地回答。"
    "\n\n"
    "文档内容:\n{context}"
)


def create_rag_chain(retriever, *, streaming=False):
    model = ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        model=LLM_MODEL,
        streaming=streaming,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    return (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
