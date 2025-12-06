from typing import Literal
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.types import Tensor
from kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load

LABELS_COLUMN = "stroke"


class StrokeDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        dataset_name = "fedesoriano/stroke-prediction-dataset"
        dataset_download(dataset_name)

        # Set the path to the file you'd like to load
        file_path = "healthcare-dataset-stroke-data.csv"
        print(f"FILE_PATH: {file_path}")

        # Load the latest version
        self.data: pd.DataFrame = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "fedesoriano/stroke-prediction-dataset",
            file_path,
        )
        assert isinstance(self.data, pd.DataFrame)

        # scaler = StandardScaler()
        # scaled_values = scaler.fit_transform(self.data)
        # self.data = pd.DataFrame(scaled_values, columns=self.data.columns, index=self.data.index)

        print("First 5 records:\n", self.data.head())
        print("levels:\n", self.data.columns)
        self.labels = self.data.loc[:, LABELS_COLUMN]

        self.data = self.data.drop(columns=LABELS_COLUMN)

    def __getitem__(self, index: Tensor | int):
        if type(index) is int:
            return Tensor(self.data.loc[index].to_numpy()), Tensor(
                self.labels[index].to_numpy()
            )
        elif type(index) is Tensor:
            return self.data.loc[index.tolist()], self.labels[index.tolist()].to_numpy()
        else:
            raise Exception("ERRO AO PEGAR DADOS")

    def __len__(self):
        return len(self.data)

    # funcao para preparacao de dados, caso seja necessario
    def data_prep(self) -> None:
        STR_COL = ["gender"]
        for col in STR_COL:
            conj_valores = set(self.data[:, col].tolist())
            i = 0
            mapeamento_final = dict()
            map(lambda x: mapeamento_final[i] = 0, conj_valores)


# class DataMappping:
#     def __init__(self, data: pd.DataFrame) -> None:
#         self.columns_map = {"gender": str, "": ""}
