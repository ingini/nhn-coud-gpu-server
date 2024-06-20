import pandas as pd

from app.vector_store_manager import VectorStoreManager
from app.db import get_db_session
from app.models import QuestionAnswer
import os
import json
from fastapi import FastAPI, HTTPException, Query
from llm import Ollama as ollama
from sqlalchemy.sql import select
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages.base import BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate,ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

app = FastAPI()
model_name = "BAAI/bge-m3"
model_kwargs = {"device": "mps"}
encode_kwargs = {"normalize_embeddings": True}
cache_dir = ".cache"
vector_manager = VectorStoreManager(model_name, model_kwargs, encode_kwargs, cache_dir)
os.makedirs(cache_dir, exist_ok=True)

def load_config():
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
    return config

async def fetch_question_answers():
    async with get_db_session() as session:
        result = await session.execute(select(QuestionAnswer))
        data = result.scalars().all()
        return [dict(question=item.question, answer=item.answer, type=item.type, link=item.link) for item in data]


def create_answer_pipeline(model, retriever, rag_prompt = None):
    prompt = ChatPromptTemplate.from_template(rag_prompt)
    return {
        "context": RunnableLambda(get_question) | retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | model | StrOutputParser()

def create_answer_pipeline2(model, retriever, rag_prompt):
    return {
        "context": RunnableLambda(get_question) | retriever | format_docs,
        "input": RunnablePassthrough(),
        "instruction": RunnablePassthrough()
    } | rag_prompt | model | StrOutputParser()

def get_question(question):
    if not question:
        return None
    elif isinstance(question,str):
        return question
    elif isinstance(question,dict) and 'question' in question:
        return question['question']
    elif isinstance(question,BaseMessage):
        return question.content
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")

def format_docs(documents):
    formatted_docs = []
    for index, doc in enumerate(documents):
        if index == 0:
            content_dict = extract_content(doc.page_content)
            doc_content = '답변: ' + content_dict.get('answer', '')
            if content_dict.get('type') == 'link' and 'link' in content_dict:
                doc_content += f"\n자세한 내용은 아래 링크를 참고해주세요: '{content_dict['link']}'"
            formatted_docs.append(doc_content)
    print(formatted_docs)
    return "\n\n".join(formatted_docs)

def extract_content(page_content):
    content_dict = {}
    parts = page_content.split(', ')
    for part in parts:
        key, value = extract_key_value(part)
        if key and value:
            content_dict[key] = value
    return content_dict

def extract_key_value(part):
    key_value = part.split(': ')
    if len(key_value) == 2:
        return key_value[0].strip(), key_value[1].strip()
    return None, None

@app.get("/create-text-index")
async def create_text_index():
    try:
        loader = TextLoader("car123.txt")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)
        vector_manager.create_index(docs)
        return {"message": "Indexing successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/create-search-index")
async def create_search_index():
    try:
        data = await fetch_question_answers()
        vector_manager.create_index(data)
        return {"message": "Indexing successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/load-vector-store")
async def load_vector_store():
    try:
        vector_manager = VectorStoreManager(model_name, model_kwargs, encode_kwargs, cache_dir)
        vector_manager.load_vector_store()
        return {"message": "Load successful"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/answer-case1")
async def answer_case1(query: str = Query(..., title="Question", description="Enter your question here:")):
    question = {"question": query}
    config = load_config()

    rag_chain = create_answer_pipeline(ollama, vector_manager.vector_store.as_retriever(), config["RAG_PROMPT_TEMPLATE"])
    try:
        response = rag_chain.invoke(question)
        print(response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.get("/answer-case2")
async def answer_case2(query: str = Query(..., title="Question", description="Enter your question here:")):
    question = {"input": query}
    config = load_config()
    prompt = PromptTemplate(template=config["RAG_PROMPT_TEMPLATE_CHAINS"], input_variables=["context", "input"])
    retriever = vector_manager.vector_store.as_retriever(search_kwargs={'k': 1})
    combine_docs_chain = create_stuff_documents_chain(ollama, prompt)
    chain = create_retrieval_chain(retriever, combine_docs_chain)
    try:
        response = chain.invoke(question)
        print(response)
    except Exception as e:
        # 기타 예외 처리
        raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

@app.get("/answer-case3")
async def answer_case3(query: str = Query(..., title="Question", description="Enter your question here:")):
    response = vector_manager.vector_store.search(query, k=1, search_type='similarity')
    print(response)

@app.get("/answer-case4")
async def answer_case4(query: str = Query(..., title="Question", description="Enter your question here:")):
    prompt_template = """Please answer according to the following REQUEST.질문 내용을 보여주고 답변을 해주세요.
    REQUEST:
    1. 안녕하세요! Turucar의 AI상담사 입니다.
    2. CONTEXT:{context}
    3. For further inquiries, please contact our customer service. Customer service: (010)-5896-5477
    ANSWER:"
    """
    prompt = PromptTemplate.from_template(prompt_template)
    rag_chain = create_answer_pipeline2(ollama, vector_manager.vector_store.as_retriever(),prompt)
    response = rag_chain.invoke({"question": query})
    print(response)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8888)