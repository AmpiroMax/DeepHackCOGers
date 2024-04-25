from typing import Optional
from pydantic import BaseModel


class Message(BaseModel):
    prompt: str = ""
    pdf_file: Optional[str] = None
