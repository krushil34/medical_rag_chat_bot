from app.components.vectore_store import load_vector_store
from app.components.llm import load_llm
from app.config.config import HUGGINGFACE_REPO_ID,HF_TOKEN
from app.common.logger import get_logger
from app.common.custom_exception import CustomException
from typing import List

from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough,RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs: List[Document]) -> str:
    """Format documents for insertion into prompt"""
    formatted = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', 'Unknown')
        formatted.append(f"Document {i+1} (Source: {source}):\n{doc.page_content}")
    return "\n\n".join(formatted)


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


logger = get_logger(__name__)

chat_promt = ChatPromptTemplate.from_template(""" Answer the following medical question in 2-3 lines maximum using only the information provided in the context.

Context: {context}

Question:{question}

Answer:
"""
)



def create_qa_chain():
    try:
        logger.info("Loading vector store for context")
        db = load_vector_store()

        if db is None:
            raise CustomException("Vector store not present or empty")

        llm = load_llm( )

        if llm is None:
            raise CustomException("LLM not loaded")
        
        retriver = db.as_retriever(search_type='similarity',search_kwargs={'k':1})
#         qa_chain = RetrievalQA.from_chain_type(
#             llm=llm,
#             chain_type="stuff",
#             retriever = db.as_retriever(search_kwargs={'k':1}),
#             return_source_documents=False,
#             chain_type_kwargs={'prompt': set_custom_prompt()}
#         )
        # simple_rag_chain=(
        #     {"context":retriever | format_docs,"question":RunnablePassthrough() }
        #     | simple_prompt
        #     | llm
        #     |StrOutputParser()

        # )
        qa_chain_lcel=(
            RunnableParallel({"context":retriver | format_docs,"question":RunnablePassthrough()})
            | chat_promt
            | llm
            |StrOutputParser()
        )

        logger.info("Sucesfully created the QA chain")
        return qa_chain_lcel
    
    except Exception as e:
        error_message = CustomException("Failed to make a QA chain", e)
        logger.error(str(error_message))



