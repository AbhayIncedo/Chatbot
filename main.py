import os
import time
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import PyPDF2
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
import requests
import pandas as pd
from os import listdir
from os.path import isfile, join
os.environ["OPENAI_API_KEY"] = "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"


def main():


    dir_path = os.path.join(os.getcwd(),"training")
    dir_path_2 = os.path.join(os.getcwd(), "EAGLE2030_Datasheet")
    print(f"HERERERERERERR == {dir_path}, +++++++ {dir_path_2}")
    pdf_path = r'/mount/src/chatbot/training'

    csv_path = r'/mount/src/chatbot/training'

    # text_path = r'/content/drive/My Drive/text'



    pdf_files = [f for f in os.listdir(pdf_path) if f.endswith('.pdf')]
    csv_files = [f for f in os.listdir(csv_path) if f.endswith(".csv")]
    # text_files = [f for f in os.listdir(text_path) if f.endswith(".txt")]

    weblinks = ["https://en.wikipedia.org/wiki/Page_(computer_memory)"]
    pdf_files = [join(pdf_path, f) for f in listdir(pdf_path) if isfile(join(pdf_path, f)) and join(pdf_path, f).endswith('.pdf')]



    for f in pdf_files:
      with open(f, 'rb') as file_handle:
        # Set strict=False to allow PDF files that don't comply to the PDF spec: https://www.pdfa.org/resource/pdf-specification-index/
        pdf_reader = PyPDF2.PdfReader(file_handle, strict=False )


        page_text = ''
        # Iterate through each page in the PDF document to extract the text and add to plain-text string
        for page_num in range(0, len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text += page.extract_text()




    csv_text = ''
    for csv_file in csv_files:
        df = pd.read_csv(os.path.join(csv_path, csv_file),encoding = 'ISO-8859-1')
        csv_text += df.to_string()


    # text_text = ''
    #
    # for text_file in text_files:
    #     with open(os.path.join(text_path, text_file), "r") as f:
    #         text_text = f.read()


    web_text = ''
    for weblink in weblinks:

        response = requests.get(weblink)
        web_text += response.text


    #print(web_text)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )


    csv_document = text_splitter.split_text(text=csv_text)
    pdf_document = text_splitter.split_text(text=page_text)
    web_document = text_splitter.split_text(text=web_text)
    # text_document = text_splitter.split_text(text=text_text)


    all_documents = csv_document + web_document  +pdf_document
    llm = ChatOpenAI(temperature=0, max_tokens=1000, model_name="gpt-3.5-turbo")
    #Vectorize the documents and create vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(all_documents, embedding=embeddings)
    #st.image('/content/MicrosoftTeams-image (11).png', width = None, use_column_width=True)
    st.title("Incedo GenAI Chatbot")
    #st.balloons()


    if csv_text and web_text  is not None:


        try:

            st.info("Type your query in the chat window.")
        except Exception as e:
            st.error(f"Error occurred while reading the PDF: {e}")
            return

        st.session_state.processed_data = {
            "document_chunks": all_documents,
            "vectorstore": vectorstore,
        }

        # Load the Langchain chatbot

        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever())

        # Initialize Streamlit chat UI
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
          with st.chat_message(message["role"]):
            st.markdown(message["content"])

        if prompt := st.chat_input("Ask your questions ?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            result = qa({"question": prompt, "chat_history": [(message["role"], message["content"]) for message in
                                                              st.session_state.messages]})
            #print(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = result["answer"]
                message_placeholder.markdown(full_response + "|")
            message_placeholder.markdown(full_response)
            #print(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})


if __name__ == "__main__":
    main()



