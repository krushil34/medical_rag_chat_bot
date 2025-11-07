from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from app.config.config import HF_TOKEN,HUGGINGFACE_REPO_ID,GROQ_TOKEN

from app.common.custom_exception import CustomException
from app.common.logger import get_logger

logger = get_logger(__name__)

def load_llm(model_name:str ="llama-3.1-8b-instant", groq_token:str = GROQ_TOKEN ):
    try:
        logger.info("Loading llm from huggingface..")
        # llm = HuggingFaceEndpoint(
        #     repo_id=huggingface_repo_id,
        #     huggingfacehub_api_token=hf_token,
        #     task="conversational",
        #     temperature=0.3,
        #     max_new_tokens=256,
        #     return_full_text=False
        # )
        llm = ChatGroq(
            api_key=groq_token,
            model= model_name,
            temperature=0.3,
            max_tokens=256
        )
        logger.info("LLM loaded successfuly..")


        return llm
    except Exception as e:
        error_message = CustomException("Failed to load a llm" , e)
        logger.error(str(error_message))