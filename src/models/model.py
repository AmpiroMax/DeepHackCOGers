from langchain.chat_models.gigachat import GigaChat
from langchain.schema import HumanMessage
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import load_prompt
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

        self.search_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
        )

        self.summarization_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=7000,
            chunk_overlap=0,
        )
        book_map_prompt = load_prompt(
            "lc://prompts/summarize/map_reduce/summarize_book_map.yaml"
        )
        book_combine_prompt = load_prompt(
            "lc://prompts/summarize/map_reduce/summarize_book_combine.yaml"
        )
        self.summarization_chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=book_map_prompt,
            combine_prompt=book_combine_prompt,
            verbose=False,
        )

        with open(prompts_config_file) as file:
            self.system_prompts = json.load(file)

    def resoner(self, prompt: str) -> COGFuncEnum:
        model_prompt = self.system_prompts["reasoner"].format(prompt)
        answer = self.llm.invoke(model_prompt).content
        command_number = re.findall(r"(?m)^(\d+).*", answer)[0]
        return COGFuncEnum(command_number)

    def chat(self, prompt: str) -> ModelAnswer:
        raise NotImplementedError

    def add_to_table(
        self, pdf_files: list[Document], fields_names: list[str]
    ) -> list[str]:
        questions = self._generate_relevant_question(fields_names)
        relevant_information = self._extract_relevant_information(pdf_files, questions)
        return relevant_information

    def get_overview(self, pdf_files: list[Document]) -> ModelAnswer:
        documents = self.summarization_text_splitter.split_documents(pdf_files)
        res = self.summarization_text_splitter.invoke({"input_documents": documents})
        answer = ModelAnswer(answer=res["output_text"])
        return answer

    def get_structured_overview(
        self, pdf_files: list[Document], prompt: str
    ) -> ModelAnswer:
        fields_names = self._extract_field_names(prompt)
        questions = self._generate_relevant_question(fields_names)
        relevant_information = self._extract_relevant_information(pdf_files, questions)

        overview = ""
        for field, info in zip(fields_names, relevant_information):
            field_info = f"------------------{field}------------------\n{info}\n\n"
            overview += field_info

        answer = ModelAnswer(answer=overview)
        return answer

    def compare_papers(self, pdf_files: list[Document], prompt: str) -> ModelAnswer:
        raise NotImplementedError

    def _extract_field_names(self, prompt: str) -> list[str]:
        model_prompt = self.system_prompts["field_extractor"].format(prompt)
        model_answer = self.llm.invoke(model_prompt).content

        fields_list = [field.split(": ")[1] for field in model_answer.split("\n")]
        print(fields_list)
        return fields_list

    def _generate_relevant_question(self, fields_names: list[str]) -> list[str]:
        question_list = []

        for field in fields_names:
            model_prompt = self.system_prompts["question_generator"].format(field)
            model_answer = self.llm.invoke(model_prompt).content

            # Recieving first question from generated questions list.
            # One may change this to summarizing all generated
            # questions or choosing random question each time.
            question = model_answer.split("\n")[0].split(". ")[1]
            question_list.append(question)
        print(question_list)
        return question_list

    def _extract_relevant_information(
        self, pdf_files: list[Document], questions: list[str]
    ) -> list[str]:
        documents = self.search_text_splitter.split_documents(pdf_files)
        db = Chroma.from_documents(
            documents,
            self.embedder,
            client_settings=Settings(anonymized_telemetry=False),
        )
        qa_chain = RetrievalQA.from_chain_type(self.llm, retriever=db.as_retriever())

        relevant_information = []

        for question in questions:
            answer = qa_chain.invoke({"query": question})["result"]
            relevant_information.append(answer)
        print(relevant_information)
        return relevant_information

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
