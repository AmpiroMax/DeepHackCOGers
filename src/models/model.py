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
from src.data.base_table import BaseTable
import json
import re


class ModelAnswer:
    output: str


class COGAPI:
    def __init__(
        self,
        auth_token: str,
        prompts_config_file: str,
        # gen_config: Optional[dict], # Controls model generation params
        scope: str = "GIGACHAT_API_CORP",
    ) -> None:
        self.llm = GigaChat(credentials=auth_token, verify_ssl_certs=False, scope=scope)

        self.embedder = GigaChatEmbeddings(
            credentials=auth_token, verify_ssl_certs=False, scope=scope
        )

        with open(prompts_config_file) as file:
            self.system_prompts = json.load(file)

    def resoner(
        self,
        prompt: str,
        pdf_files: Optional[list[Document] | Document],
        table: BaseTable,
    ) -> Optional[ModelAnswer]:

        model_prompt = self.system_prompts["reasoner"].format(prompt)
        answer = self.llm.invoke(model_prompt).content
        command_number = re.findall(r"(?m)^(\d+).*", answer)[0]

        if command_number == 0:
            self.add_to_table

    def chat(self) -> ModelAnswer:
        pass

    def add_to_table(self, table: BaseTable, pdf_file: Document) -> None:
        pass

    def get_overview(self, pdf_file: Document) -> ModelAnswer:
        pass

    def get_structured_overview(self, pdf_file: Document, prompt: str) -> ModelAnswer:
        pass

    def compare_papers(self, pdf_files: list[Document], prompt: str) -> ModelAnswer:
        pass
