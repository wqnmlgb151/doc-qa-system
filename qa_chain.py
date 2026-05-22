from functools import lru_cache
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.runnables import Runnable

from config import DASHSCOPE_API_KEY, BASE_URL, LLM_MODEL


def format_docs(docs: List[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


SYSTEM_PROMPT = (
    "你是一个基于文档内容进行问答的助手。"
    "请使用以下检索到的文档内容来回答用户的问题。"
    "如果你无法从文档中找到答案，请诚实地说明你不知道，"
    "不要编造信息。请尽量用八句话以内简洁地回答。"
    "\n\n"
    "文档内容:\n{context}"
)


@lru_cache(maxsize=1)
def _get_streaming_model() -> ChatOpenAI:
    return ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        model=LLM_MODEL,
        streaming=True,
    )


def create_chain_from_docs(docs: List[Document], *, streaming: bool = False) -> Runnable:
    """Create a RAG chain using pre-retrieved (and optionally pre-reranked) documents.

    Retrieval and reranking are handled by the caller so the route can
    inject degradation warnings into the SSE stream before generation.
    """
    model = _get_streaming_model() if streaming else ChatOpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=BASE_URL,
        model=LLM_MODEL,
        streaming=False,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    context_str = format_docs(docs)

    return (
        RunnablePassthrough()
        | (lambda q: {"context": context_str, "input": q})
        | prompt
        | model
        | StrOutputParser()
    )
