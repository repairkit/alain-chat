alain-chat 

アランの幸福論を基にしたRAGを作り、チャットボットに幸福論に基づいたアドバイスをもらうアプリ

開発はUbuntuで行いました

技術
langchain
fastapi(バックエンドのapi処理)
streamlit(demoのフロントエンド)

React vite (フロントエンド) alain-chat-frontendリポジトリを参照

.envに OPENAI_API_KEY = "xxx" を設定してください

起動方法(server.py)
1.Dockerをビルド
docker build -t alain-chat .
2.Dockerを起動
docker run -p 8080:8080 alain-chat

バックエンドを起動した状態で、フロントエンド(alain-chat-frontend)も起動してください

demo起動方法(chat_streamlit.py)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd /demo
streamlit run chat_streamlit.py

chat.py
アランの幸福論の原文（フランス語）をベクトルデータベース化し、RAGを作成した。質問クエリは日本語であると想定し、クエリを翻訳->検索->日本語で生成のプロセスにした。

次の開発目標
前回のやり取りが記憶されていないので、historyを作成してプロンプトに加える。ただ、トークン量によっては前何回だけにするとか、要約などの機能を付けたい。
