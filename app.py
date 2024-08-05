import os
from dotenv import load_dotenv
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI


def main(url: str) -> None:
    # definindo para utilizar o novo modelo da OpenAI
    llm = OpenAI(model='gpt-4o-mini')

    # gerando os documentos a partir da url
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls=[url])

    # criando o armazenamento de vetores
    index = VectorStoreIndex.from_documents(documents=documents)

    # mecanismo de consulta no index
    query_engine = index.as_query_engine(llm=llm)

    # realizando uma query
    response = query_engine.query("O que é um Sistema Operacional?")
    print(response)


if __name__ == '__main__':
    # carregando envs
    load_dotenv()
    print("Hello World Llamaindex Course")

    # imprimindo chave da openai
    print(f'OPENAI_API_KEY {os.environ['OPENAI_API_KEY']}')
    print('*****************')

    # artigo utilizado para fazer perguntas
    main(url='https://medium.com/com-br/o-que-%C3%A9-um-sistema-operacional-bc31756926cc')