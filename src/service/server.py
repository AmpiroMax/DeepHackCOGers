from src.models.model import COGAgent, AgentAnswer
from src.shemas.messages import Message
from src.shemas.enums import COGFuncEnum
from src.data.base_table import BaseTable
from typing import Optional


class COGServer:
    def __init__(
        self,
        auth_token: str,
        prompts_config_file: str,
    ) -> None:
        self.agent = COGAgent(auth_token, prompts_config_file)
        self.table = None

    def set_table(self, table: BaseTable) -> None:
        self.table = table

    def get_table(self) -> BaseTable:
        return self.table

    def add_task(self, message: Message) -> AgentAnswer:
        # In the future one may add message queue.
        # Also it would be great to rewrite part of
        # request-response code in the async manner.
        answer: AgentAnswer = self._call_agent(message)

        if not answer.info is None:
            self.table.add_paper_to_table(answer.info)

        return answer

    def _call_agent(self, message: Message) -> AgentAnswer:
        try:
            func: COGFuncEnum = self.agent.reasoner(message.prompt)
        except Exception as e:
            print(f"During reasoning exception was raised. {e=}")
            answer = AgentAnswer(
                answer=f"During model call with '{func.name}' smth went wrong. Exception was raised. {e=}"
            )
            return answer

        answer: Optional[AgentAnswer] = None

        # This section could be written with switch or mapper dict statments
        try:
            if func == COGFuncEnum.CHAT:
                answer = self.agent.chat(message.prompt)

            if func == COGFuncEnum.OVERVIEW:
                answer = self.agent.get_overview(message.pdf_files)

            if func == COGFuncEnum.STRUCTURED_OVERVIEW:
                answer = self.agent.get_structured_overview(
                    message.pdf_files, message.prompt
                )

            if func == COGFuncEnum.COMPARE:
                answer = self.agent.compare_papers(message.pdf_files, message.prompt)

            if func == COGFuncEnum.ADD_TO_TABLE:
                field_names = self.table.get_headers()
                answer = self.agent.add_to_table(message.pdf_files, field_names)

        except Exception as e:
            print(f"During model call with '{func.name}' exception was raised. {e=}")
            answer = None

        if answer is None:
            answer = AgentAnswer(
                answer=f"During model call with '{func.name}' smth went wrong."
            )

        return answer
