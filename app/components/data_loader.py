import os
from app.components.pdf_loader import load_pdf_files,create_text_chunks
from app.components.vectore_store import create_save_vectorestore
from app.config.config import DB_FAISS_PATH

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

logger= get_logger(__name__)

def process_and_store_pdfs():
    try:
        logger.info("making the vectorstore...")
        documents=load_pdf_files()
        text_chunks = create_text_chunks(documents)
        create_save_vectorestore(text_chunks)
        logger.info("Vectorstore created sucesfully....")

    except Exception as e:
        error_message = CustomException("Failed to create vectore database",e)
        logger.error(str(error_message))


    if __name__=="__main__":
        process_and_store_pdfs()