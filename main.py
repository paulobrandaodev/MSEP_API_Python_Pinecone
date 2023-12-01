import os
from flask import Flask, request
import pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings import OpenAIEmbeddings

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = Flask(__name__)

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment='gcp-starter'
)
index_pinecone = 'jarvis'


@app.route("/", methods=["POST"])
def search():
    question = request.json["question"]

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

    docsearch = Pinecone.from_existing_index(index_pinecone, embeddings)

    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = docsearch.similarity_search(question)

    return {"resposta": chain.run(input_documents=docs, question=question)}


if __name__ == "__main__":
    app.run(debug=True)
