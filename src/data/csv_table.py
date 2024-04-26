from src.data.base_table import BaseTable
import pandas as pd


class CsvTable(BaseTable):
    def __init__(self, table_path: str = None, pd_table: pd.DataFrame = None) -> None:
        super().__init__()
        if not table_path is None:
            self.table = pd.read_csv(table_path)
        else:
            self.table = pd_table

    def save_table(self, save_path: str) -> None:
        self.table.to_csv(save_path)

    def get_headers(self) -> list[str]:
        return list(self.table.columns)

    def get_table(self) -> pd.DataFrame:
        return self.table

    def add_paper_to_table(self, data_to_add: list[str]) -> None:
        self.table.loc[len(self.table.index)] = data_to_add


if __name__ == "__main__":
    csv_path = "/Users/ampiro/programs/HACKATONS/DeepHackCOGers/data/raw/table.csv"
    my_table = CsvTable(csv_path)

    print(my_table.get_headers())
