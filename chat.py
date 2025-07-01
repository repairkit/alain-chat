from dotenv import load_dotenv
load_dotenv()
from langchain_unstructured.document_loaders import UnstructuredLoader
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

import langchain
import os
file_path = "alain_propos_bonheur.doc"
loader = UnstructuredLoader(
    file_path=file_path,
)
# ベクトルデータベースが対応していないメタデータの削除
documents = filter_complex_metadata(loader.load())
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  # チャンクサイズ
    chunk_overlap=200  # オーバーラップ
)
documents = text_splitter.split_documents(documents)
#チャンク確認
def analysis_langchain_documents(documents, break_point=10):
    num_documents = len(documents)
    lengths = [len(doc.page_content) for doc in documents]
    total_length = sum(lengths)

    print(f"Number of documents: {num_documents}")
    print(f"Total length of text: {total_length} characters")
    print(f'Individual document lengths: {lengths[:10]} {"" if len(lengths) <= 10 else "..."}')

    for i, doc in enumerate(documents):
        print("="*10)
        print(f"Document {i+1} (Length: {len(doc.page_content)}): {doc.page_content[:100]}...")
        print("="*10+"\n\n")
        if i >= break_point:
            break

#VectorDBの定義
class VectorDB:
    def __init__(self, db_name):
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')
        self.db = Chroma(persist_directory=db_name, embedding_function=self.embeddings)

    def add_documents(self, documents):
        self.db.add_documents(documents)

    def similarity_search(self, query, **kwargs):
        return self.db.similarity_search(query, **kwargs)
       
    def as_retriever(self, **kwargs):
        return self.db.as_retriever(**kwargs)
    
    
def make_vector_db():
    vector_db = VectorDB(db_name='vector_db')
    vector_db.add_documents(documents)


def create_multilingual_rag_chain(vector_db):
    # 質問翻訳用のチェーン
    translation_prompt = ChatPromptTemplate.from_template('''
    以下の日本語の質問をフランス語に翻訳してください。翻訳結果のみを返してください。
    
    質問: {question}
    ''')
    
    # 回答生成用のプロンプト
    answer_prompt = ChatPromptTemplate.from_template('''
    以下のフランス語の情報源を参考にして、日本語で質問に答えてください。
    
    元の質問（日本語）: {original_question}
    
    情報源（フランス語）:
    {context}
    
    回答は日本語で、情報源の内容を正確に反映してください。
    ''')

    llm = ChatOpenAI(model='gpt-4o-mini', temperature=0.1)
    retriever = vector_db.as_retriever(search_kwargs={"k": 1}) #うまく分割できているため、1つの情報源で十分

    # シンプルな修正版
    chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            translated_question=translation_prompt | llm | StrOutputParser()
        )
        | RunnablePassthrough.assign(
            original_question=lambda x: x["question"],
            context=lambda x: retriever.invoke(x["translated_question"])
        )
        | answer_prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

if __name__ == "__main__":
    vector_db = VectorDB('vector_db')
    
    # ディレクトリの存在チェック（より確実）
    if not os.path.exists('vector_db') or len(os.listdir('vector_db')) == 0:
        print("Vector DB not found or empty, creating...")
        vector_db.add_documents(documents) 
    else:
        print("Vector DB found, using existing data.")
    
    question = '不安になったらどうすればいいですか？'
    results = vector_db.similarity_search(question, k=2)  
    print("searched results: " + str([result.page_content for result in results]))
    chain = create_multilingual_rag_chain(vector_db)
    result = chain.invoke(question)
    print("回答:" + result)