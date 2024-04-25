from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings
from langchain.vectorstores import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from typing import Optional
from pydantic import BaseModel

from src.data.base_table import BaseTable
from src.shemas.enums import COGFuncEnum
import json
import re


class ModelAnswer(BaseModel):
    answer: str


class COGAPI:
    def __init__(
        self,
        auth_token: str,
        prompts_config_file: str,
        scope: str = "GIGACHAT_API_CORP",
    ) -> None:
        self.llm = GigaChat(credentials=auth_token, verify_ssl_certs=False, scope=scope)

        self.embedder = GigaChatEmbeddings(
            credentials=auth_token, verify_ssl_certs=False, scope=scope
        )

        with open(prompts_config_file) as file:
            self.system_prompts = json.load(file)

    def resoner(self, prompt: str) -> COGFuncEnum:
        model_prompt = self.system_prompts["reasoner"].format(prompt)
        answer = self.llm.invoke(model_prompt).content
        command_number = re.findall(r"(?m)^(\d+).*", answer)[0]
        return COGFuncEnum(command_number)

    def chat(self, prompt: str) -> ModelAnswer:
        pass

    def add_to_table(self, pdf_file: Document, fields_names: list[str]) -> list[str]:
        pass

    def get_overview(self, pdf_file: Document) -> ModelAnswer:
        pass

    def get_structured_overview(self, pdf_file: Document, prompt: str) -> ModelAnswer:
        pass

    def compare_papers(self, pdf_files: list[Document], prompt: str) -> ModelAnswer:
        pass

    def _extract_field_names(self, prompt: str) -> list[str]:
        pass

    def _generate_relevant_question(self, fields_names: list[str]) -> list[str]:
        pass

    def _extract_relevant_information(
        self, pdf_file: Document, questions: list[str]
    ) -> list[str]:
        pass

    def _success_message(self) -> ModelAnswer:
        message = ModelAnswer(
            answer=self.llm.invoke(self.system_prompts["success_message"].content)
        )
        return message

    def _error_message(self) -> ModelAnswer:
        message = ModelAnswer(
            answer=self.llm.invoke(self.system_prompts["error_message"].content)
        )
        return message
