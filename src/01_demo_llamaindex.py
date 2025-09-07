from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

# 读取文档
documents = SimpleDirectoryReader('docs').load_data()

# 构建索引
index = VectorStoreIndex.from_documents(documents)

# 创建查询引擎
query_engine = index.as_query_engine()

# 查询
response = query_engine.query('')
print(response)