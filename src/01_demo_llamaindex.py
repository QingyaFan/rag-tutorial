from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama


def answer_with_rag(question: str):
    # 读取文档
    documents = SimpleDirectoryReader('docs').load_data()
    # 构建索引
    index = VectorStoreIndex.from_documents(documents)
    # 创建查询引擎
    query_engine = index.as_query_engine(
        retriever_mode='hybrid',
        similarity_top_k=3,
        retriever_kwargs={'similarity_top_k': 3}
    )
    print(query_engine.query(question))


def answer_without_rag(question: str):
    llm = Ollama(model='qwen3:0.6b', request_timeout=120)
    response = llm.complete(question)
    print(response.text)

if __name__ == '__main__':
    question = '请用中文回答：请描述一下你的工作环境。'
    answer_without_rag(question)