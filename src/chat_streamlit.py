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
    あなたはフランスの哲学者アランです。あなた自身の思想、特に『幸福論』に基づいて、ユーザーの悩みや問いに答えてください。
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

def chat(question):
    vector_db = VectorDB('vector_db')
    
    # ディレクトリの存在チェック（より確実）
    if not os.path.exists('vector_db') or len(os.listdir('vector_db')) == 0:
        print("Vector DB not found or empty, creating...")
        vector_db.add_documents(documents) 
    else:
        print("Vector DB found, using existing data.")
    
    results = vector_db.similarity_search(question, k=2)  
    print("searched results: " + str([result.page_content for result in results]))
    chain = create_multilingual_rag_chain(vector_db)
    result = chain.invoke(question)
    return result

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

if __name__ == "__main__":
    result = chat('不安になったらどうすればいいですか？')
    print("回答:" + result)
    
    
#ここからStreamlitアプリのコード
import streamlit as st

# アプリのタイトルを設定
st.title("Alain Chat")

# タイトルの下に説明を追加
st.markdown("""
### アランの幸福論について質問できるチャットボット

このチャットボットは、アランの「幸福論」に基づいて質問に答えます。
不安や幸福について、アランの哲学的な視点から回答を提供します。

**質問例:**
- 不安になったらどうすればいいですか？
- 幸せになるためには何が必要ですか？
- 悲しい気持ちを乗り越える方法は？
""")

# 区切り線を追加（オプション）
st.divider()

# st.session_stateにメッセージの履歴を保存するリストを初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# 過去のメッセージを全て表示
# st.chat_message を使うと、役割（"user"や"assistant"）に応じたアイコンで表示される
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# st.chat_inputでユーザーからの入力を受け付ける
# user_inputにユーザーが入力した文字列が格納される
if user_input := st.chat_input("メッセージを入力してください。"):
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # ユーザーの入力をボットに渡して応答を取得
    # chat関数は、質問を受け取り、アランの哲学に基づいて応答を生成する
    bot_response = chat(user_input)
    
    # ボットの応答を履歴に追加して表示
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # 必要に応じてst.rerun()を呼び出して画面を更新することもできますが、
    # Streamlitの実行モデルでは通常、ウィジェット操作後に自動でスクリプトが再実行されるため、
    # このシンプルなケースでは明示的な呼び出しは不要です。