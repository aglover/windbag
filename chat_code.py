from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import Language

class ChatCode:
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="codellama")
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] You are an expert programmer for question-answering tasks. Use the following pieces of retrieved context 
            to answer the question. If you don't know the answer, just say that you don't know. Use three sentences
            maximum and keep the answer concise. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, path: str):
        
        loader = GenericLoader.from_filesystem(
            path,
            glob="**/[!.]*",
            suffixes=[".rb"],
            parser=LanguageParser(),
        )

        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.RUBY, chunk_size=1024, chunk_overlap=100)

        chunks = text_splitter.split_documents(loader.load())
        chunks = filter_complex_metadata(chunks)

        vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3,"score_threshold": 0.5}
        )

        self.chain = ({"context": retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a directory first."

        return self.chain.invoke(query)

    def clear(self):
        self.chain = None
