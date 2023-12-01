import os
from flask import Flask, request
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings

# Carrega as chaves de API do ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Cria a aplicação Flask
app = Flask(__name__)

# Inicia a conexão com o Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment='gcp-starter'
)

# Define o nome do índice Pinecone
index_pinecone = 'jarvis'


@app.route("/", methods=["POST"])
def search():
    # Extrai a pergunta da requisição JSON
    question = request.json["question"]

    # Cria um objeto OpenAIEmbeddings para gerar embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    # Cria um objeto Pinecone a partir do índice existente
    docsearch = Pinecone.from_existing_index(index_pinecone, embeddings)

    # Instancia o modelo ChatOpenAI e a cadeia de QA
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    # Busca os documentos semelhantes à pergunta
    docs = docsearch.similarity_search(question)

    # Executa a cadeia de QA com os documentos recuperados
    resposta = chain.run(input_documents=docs, question=question)

    # Retorna a resposta da cadeia de QA
    return {"resposta": resposta}


if __name__ == "__main__":
    app.run(debug=True)
