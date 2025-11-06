from langchain_community.vectorstores import FAISS
from app.components.embedding import get_embedding_model

import os
from app.config.config import DB_FAISS_PATH
from app.common.logger import get_logger
from app.common.custom_exception import CustomException

logger  = get_logger(__name__)

# Load Existing Vectorstore
def load_vector_store():
    try:
        embedding_model = get_embedding_model()
        if os.path.exists(DB_FAISS_PATH):
            logger.info("Loading Existing Vectorstore..")
            return FAISS.load_local(
                DB_FAISS_PATH,
                embeddings=embedding_model,
                allow_dangerous_deserialization=True
            )
        else :
            logger.warning("No Existing Vectore store FOund")

    except Exception as e:
        error_message = CustomException("Failed to load vector store", e)
        logger.error(str(error_message))
        raise error_message
    

# Creating New VectorStore
def create_save_vectorestore(text_chunks):
    try:
        if not text_chunks:
            raise CustomException("No text chunks found..")
        
        logger.info("Generating new Vectorstore")
        embedding_model = get_embedding_model()
        db = FAISS.from_documents(
            text_chunks,
            embedding= embedding_model
        )

        logger.info("Saving Vectorstore")

        db.save_local(DB_FAISS_PATH)
        logger.info("Vectorstore saved succesfully")

        return db
    except Exception as e:
        error_message = CustomException("Error accured while creating vector store", e)
        logger.error(str(error_message))
        raise error_message
    