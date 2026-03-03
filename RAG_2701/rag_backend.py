#1 import OS , document loader, text splitter, bedrock embeddings, vector DB, VectorStoreIndex, bedrock LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.indexes import VectorstoreIndexCreator
from langchain_aws import BedrockLLM
from langchain_aws import ChatBedrock

#5c wrap this inside a func
def hr_index():

    #2 define the data source and load data with PDF loader
    data_load = PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
    # data_test = data_load.load_and_split()
    # print(len(data_test))
    # print(data_test[0])

    #3 split the text based character, tokens etc. - Recursively split by character
    data_split = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
    # data_sample = 'The objective is to provide information to all the employees about the leaves and holidays followed in UPL India. Employees need adequate time to celebrate festival holidays, rest and recuperate and spend quality time with family and friends. This policy is effective from 1st October 2020.'
    # data_split_test = data_split.split_text(data_sample)
    # print(data_split_test)


    #4 Create Embeddings - client connection
    data_embeddings = BedrockEmbeddings(
        credentials_profile_name='default',
        model_id="amazon.titan-embed-text-v2:0"
    )

    #5 create vector DB, store embeddings and index for search - VectorStoreIndexCreator
    data_index = VectorstoreIndexCreator(
        text_splitter=data_split,
        embedding=data_embeddings,
        vectorstore_cls=FAISS
    )

    #5 create index for HR policy document
    db_index = data_index.from_loaders([data_load])
    return db_index

#6 write a function to connect to the bedrock foundation model and 
def hr_llm():
    llm=ChatBedrock(
        credentials_profile_name='default',
        model_id="amazon.nova-pro-v1:0",
        model_kwargs={
            "max_tokens_to_sample": 300,
            "temperature": 0.1,
            "top_p": 0.9
        })
    return llm

#6b write a function which searches the user prompt, searches the best match from the vector DB and sends both to LLM
def hr_rag_response(index, question):
    rag_llm = hr_llm()
    hr_rag_query = index.query(question= question, llm= rag_llm)
    return hr_rag_query


