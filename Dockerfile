# Python 3.12のベースイメージを使用
FROM python:3.12-slim

# 作業ディレクトリを設定
WORKDIR /app

# システムパッケージのアップデートと必要な依存関係をインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    libmagic-dev \
    libreoffice \
    && rm -rf /var/lib/apt/lists/*

# requirements.txtをコピーして依存関係をインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルをコピー
COPY src/ ./src/
COPY vector_db/ ./vector_db/
COPY alain_propos_bonheur.doc .
COPY .env .env

# PYTHONPATHを設定してsrcディレクトリをモジュール検索パスに追加
ENV PYTHONPATH=/app/src

# ポート8080を公開
EXPOSE 8080

# サーバーを起動（srcディレクトリのserver.pyを実行）
CMD ["python", "src/server.py"]
