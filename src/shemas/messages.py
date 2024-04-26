from typing import Optional
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class Message:
    prompt: str = ""
    pdf_files: Optional[list[Document]] = None
