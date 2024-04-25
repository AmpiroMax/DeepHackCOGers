from src.data.base_table import BaseTable
import pandas as pd


class CsvTable(BaseTable):
    def __init__(self, table_path: str) -> None:
        super().__init__()
        self.table = pd.read_csv(table_path)

    def save_table(self, save_path: str) -> None:
        self.table.to_csv(save_path)

    def get_headers(self) -> list[str]:
        return list(self.table.columns)

    def get_table(self) -> pd.DataFrame:
        return self.table


if __name__ == "__main__":
    csv_path = "/Users/ampiro/programs/HACKATONS/DeepHackCOGers/data/raw/table.csv"
    my_table = CsvTable(csv_path)

    print(my_table.get_headers())
