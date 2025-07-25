from torch.utils.data import Dataset
import duckdb
import torch

class DuckDBJSONDataset(Dataset):
    """Dataset backed by DuckDB for JSON or NDJSON files."""
    def __init__(self, json_path: str, feature_cols: list[str], label_col: str, query: str | None = None, transform=None):
        self.con = duckdb.connect(database=":memory:")
        self.con.execute(
            f"""
            CREATE TABLE data AS
            SELECT * FROM read_json_auto('{json_path}')
            """
        )
        sql = query or "SELECT * FROM data"
        df = self.con.execute(sql).df()
        self.X = df[feature_cols].to_numpy(dtype=float)
        self.y = df[label_col].to_numpy(dtype=int)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        x = self.X[idx]
        y = self.y[idx]
        if self.transform:
            x = self.transform(x)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
