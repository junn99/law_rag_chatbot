from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import load_prompt
from langchain_teddynote.korean import stopwords
import os
from langchain_teddynote.community.pinecone import create_index
from langchain_teddynote.community.pinecone import load_sparse_encoder
from langchain_teddynote.community.pinecone import init_pinecone_index
from langchain_teddynote.community.pinecone import PineconeKiwiHybridRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 임베딩(Embedding) 생성
model_name = "BAAI/bge-m3"
model_kwargs = {"device":"cpu"}
encode_kwargs = {"normalize_embeddings":True}
embeddings = HuggingFaceBgeEmbeddings(
model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs

)


class PDFRAG: # 데모용 : 바로 만들어서 확인하는 용도
    def __init__(self, file_path: str, llm):
        self.file_path = file_path
        self.llm = llm

    def load_documents(self):
        # 문서 로드(Load Documents)
        loader = PDFPlumberLoader(self.file_path)
        docs = loader.load()
        return docs

    def split_documents(self, docs):
        # 문서 분할(Split Documents)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_documents = text_splitter.split_documents(docs)
        return split_documents

    def create_vectorstore(self, split_documents):
        

        # DB 생성(Create DB) 및 저장
        vectorstore = FAISS.from_documents(
            documents=split_documents, embedding=embeddings
        )
        return vectorstore

    def create_retriever(self):
        vectorstore = self.create_vectorstore(
            self.split_documents(self.load_documents())
        )
        # 검색기(Retriever) 생성
        retriever = vectorstore.as_retriever()
        return retriever

    def create_chain(self, retriever):
        # 프롬프트 생성(Create Prompt)
        prompt = PromptTemplate.from_template(
            """You are a legal counselor. 
            Give your best thoughtful answer in Korean using the given context.
            Even if the given context isn't relevant to the question, try to relate it as best you can.
            Let's think step by step
        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        # 체인(Chain) 생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    
class CHROMARAG: # 기존의 ChromaDB 인덱스를 가져와서 바로 실행
    def __init__(self, index_path: str, llm):
        self.index_path = index_path
        self.llm = llm

    def bring_vectorstore(self):
        

        vectorstore = Chroma(
            persist_directory=self.index_path,
            embedding_function=embeddings,
            collection_name="Low_chroma_300chunk_db" # Low_chroma_300chunk_db, Low_chroma_db
        )

        return vectorstore

    def create_retriever(self):
        vectorstore = self.bring_vectorstore()
        # 검색기(Retriever) 생성
        retriever = vectorstore.as_retriever()
        return retriever

    def create_chain(self, retriever):
        # 프롬프트 생성(Create Prompt)
        prompt = PromptTemplate.from_template(
            """You are a legal counselor. 
            Give your best thoughtful answer in Korean using the given context.
            Even if the given context isn't relevant to the question, try to relate it as best you can.
            Let's think step by step
            Speak Korean

        #Context: 
        {context}

        #Question:
        {question}

        #Answer:"""
        )

        # 체인(Chain) 생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain


class CHROMARAG_PROMPT_YAML: # 기존의 ChromaDB 인덱스를 가져와서 바로 실행 +  + prompt 따로 관리
    def __init__(self, index_path: str, llm):
        self.index_path = index_path
        self.llm = llm

    def bring_vectorstore(self):
        

        vectorstore = Chroma(
            persist_directory=self.index_path,
            embedding_function=embeddings,
            collection_name="Low_chroma_300chunk_db" # Low_chroma_300chunk_db, Low_chroma_db
        )

        return vectorstore

    def create_retriever(self):
        vectorstore = self.bring_vectorstore()
        # 검색기(Retriever) 생성
        retriever = vectorstore.as_retriever()
        return retriever

    def create_chain(self, retriever):
        # 프롬프트 생성(Create Prompt)
        prompt = load_prompt("prompts/legal_cot.yaml") # 만약에 오류나면 encoding="utf-8" 추가

        # 체인(Chain) 생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain



class PINECONERAG: # 기존의 ChromaDB 인덱스를 가져와서 바로 실행 +  + prompt 따로 관리
    def __init__(self,llm):
        self.llm = llm

    def create_retriever(self):

        model_name = "BAAI/bge-m3"
        model_kwargs = {"device":"cpu"}
        encode_kwargs = {"normalize_embeddings":True}
        embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs

        )

        pinecone_params = init_pinecone_index(
            index_name="law-db-index",  # Pinecone 인덱스 이름
            namespace="law-pdf-hypersearch_rag",  # Pinecone Namespace
            api_key=os.environ["PINECONE_API_KEY"],  # Pinecone API Key
            sparse_encoder_path="/home/jun/my_project/Law_RAG_PJ/test/sparse_encoder.pkl", # "./sparse_encoder.pkl",  # Sparse Encoder 저장경로(save_path)
            stopwords=stopwords(),  # 불용어 사전
            tokenizer="kiwi",
            embeddings=embeddings,  # Dense Embedder
            top_k=5,  # Top-K 문서 반환 개수
            alpha=0.75,  # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
                # 처음 실험 0.5 -> 0.75로 변경
                # sem_score는 오히려 떨어짐
        )

        pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)
        return pinecone_retriever


    def create_chain(self, retriever):
        # 프롬프트 생성(Create Prompt)
        prompt = load_prompt("/home/jun/my_project/Law_RAG_PJ/prompts/legal_cot.yaml") # 만약에 오류나면 encoding="utf-8" 추가

        # 체인(Chain) 생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain
    


class PineconeRAG_Rerank: # Pinecone + HybridSearch(7:3) + prompt_load + Reranking
    def __init__(self,llm):
        self.llm = llm

    def create_retriever(self):

        model_name = "BAAI/bge-m3"
        model_kwargs = {"device":"cpu"}
        encode_kwargs = {"normalize_embeddings":True}
        embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs

        )

        pinecone_params = init_pinecone_index(
            index_name="law-db-index",  # Pinecone 인덱스 이름
            namespace="law-pdf-hypersearch_rag",  # Pinecone Namespace
            api_key=os.environ["PINECONE_API_KEY"],  # Pinecone API Key
            sparse_encoder_path="/home/jun/my_project/Law_RAG_PJ/test/sparse_encoder.pkl", # "./sparse_encoder.pkl",  # Sparse Encoder 저장경로(save_path)
            stopwords=stopwords(),  # 불용어 사전
            tokenizer="kiwi",
            embeddings=embeddings,  # Dense Embedder
            top_k=5,  # Top-K 문서 반환 개수
            alpha=0.5,  # alpha=0.75로 설정한 경우, (0.75: Dense Embedding, 0.25: Sparse Embedding)
        )

        pinecone_retriever = PineconeKiwiHybridRetriever(**pinecone_params)

        model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
        # 상위 3개의 문서 선택
        compressor = CrossEncoderReranker(model=model, top_n=4)
        # 문서 압축 검색기 초기화
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=pinecone_retriever
        )
        return compression_retriever


    def create_chain(self, retriever):
        # 프롬프트 생성(Create Prompt)
        prompt = load_prompt("/home/jun/my_project/Law_RAG_PJ/prompts/legal_cot.yaml") # 만약에 오류나면 encoding="utf-8" 추가

        # 체인(Chain) 생성
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain