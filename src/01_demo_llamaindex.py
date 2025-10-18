from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


ollama_llm = Ollama(
    model='qwen3:0.6b',
    base_url="http://localhost:11434",
    request_timeout=60
)
ollama_embed_model = OllamaEmbedding(
    model_name='bge-m3',
    base_url="http://localhost:11434"
)
Settings.llm = ollama_llm
Settings.embed_model = ollama_embed_model

def answer_with_rag(question: str):
    # 读取文档
    documents = SimpleDirectoryReader(
        input_files=['./docs/website/content/zh-cn/docs/concepts/overview/components.md']
    ).load_data()
    # 构建索引
    index = VectorStoreIndex.from_documents(documents)
    # 创建查询引擎
    query_engine = index.as_query_engine()
    response = query_engine.query(question)
    print(response.response)


def answer_without_rag(question: str):
    response = ollama_llm.complete(question)
    print(response.text)

if __name__ == '__main__':
    question = 'k8s有那些核心组件'
    # answer_without_rag(question)
    answer_with_rag(question)