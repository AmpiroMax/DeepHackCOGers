from abc import ABC


class BaseTable(ABC):
    def save_table(self) -> None:
        raise NotImplementedError

    def add_paper_to_table(self, data_to_add: list[str]) -> None:
        raise NotImplementedError

    def get_headers(self) -> None:
        raise NotImplementedError

    def get_table(self) -> None:
        raise NotImplementedError
