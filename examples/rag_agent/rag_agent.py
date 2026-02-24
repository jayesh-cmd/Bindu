import os
from bindu.penguin.bindufy import bindufy
from agno.agent import Agent
from agno.models.openrouter import OpenRouter

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

emb_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

agent = Agent(
    instructions="Answer questions using retrieved context only.",
    model=OpenRouter(
        id="openai/gpt-oss-120b",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    ),
)

def run_rag(question: str):
    data = """
    LangChain is a framework for developing applications powered by language models.
    """

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(data)

    docs = [Document(page_content=c) for c in chunks]
    vectorstore = FAISS.from_documents(docs, emb_model)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    retrieved_docs = retriever.invoke(question)

    return retrieved_docs[0].page_content


config = {
    "name": "rag_agent",
    "description": "RAG question answering agent",
    "skills": [],
}


def handler(messages):
    question = messages[-1]["content"]

    context = run_rag(question)

    result = agent.run(
        input=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:{question}"
        }]
    )

    return result


bindufy(config, handler)