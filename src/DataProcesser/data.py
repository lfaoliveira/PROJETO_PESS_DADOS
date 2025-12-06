from enum import StrEnum

from torch.utils.data import Dataset
from pandas import Series, DataFrame
import pandera.pandas as pa
from pandera.typing import DataFrame as PaDataFrame, Series as PaSeries
from sklearn.preprocessing import StandardScaler
from torch.types import Tensor
from kagglehub import KaggleDatasetAdapter, dataset_download, dataset_load

LABELS_COLUMN = "stroke"


class CATEGORICAL_COLUMNS(StrEnum):
    GENDER = "gender"
    MARRIED = "ever_married"
    WORK = "work_type"
    RESIDENCE = "Residence_type"
    SMOKE_STATUS = "smoking_status"


class MySchema(pa.DataFrameModel):
    id: PaSeries[int]
    age: PaSeries[int]
    gender: PaSeries[str]
    ever_married: PaSeries[str]
    work_type: PaSeries[str]
    Residence_type: PaSeries[str]
    smoking_status: PaSeries[str]
    hypertension: PaSeries[int]
    heart_disease: PaSeries[int]
    avg_glucose_level: PaSeries[float]
    bmi: PaSeries[float]
    stroke: PaSeries[float]



class StrokeDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()

        dataset_name = "fedesoriano/stroke-prediction-dataset"
        dataset_download(dataset_name)
        # Set the path to the file you'd like to load
        file_path = "healthcare-dataset-stroke-data.csv"
        print(f"FILE_PATH: {file_path}")

        # Load the latest version
        self.data: DataFrame = dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "fedesoriano/stroke-prediction-dataset",
            file_path,
        )
        print(f"DF NORMAL: {self.data.head()}\n")

        assert isinstance(self.data, DataFrame)
        STR_COL = list(CATEGORICAL_COLUMNS)
        ## AQUI VAI PREPARACAO DOS DADOS
        self.data_prep(STR_COL)

        print(f"levels:\n {self.data.columns}\n")
        self.labels = self.data.loc[:, LABELS_COLUMN]

        self.data = self.data.drop(columns=LABELS_COLUMN)

    def __getitem__(self, index: Tensor | int):
        if type(index) is int:
            return Tensor(self.data.loc[index].to_numpy()), Tensor(
                [self.labels[index]]
            ) 
        elif type(index) is Tensor:
            return self.data.loc[index.tolist()], self.labels[index.tolist()].to_numpy()
        else:
            raise Exception("ERRO AO PEGAR DADOS")

    def __len__(self):
        return len(self.data)

    # funcao para preparacao de dados, caso seja necessario
    def data_prep(self, bad_columns: list[CATEGORICAL_COLUMNS]) -> None:
        STR_COL = bad_columns

        ##itera sobre conjunto da coluna e bota numero pra cada string
        for col in STR_COL:
            self.data[col] = self.data[f"{col}"].astype("category")
            self.data[f"{col}_code"] = self.data[f"{col}"].cat.codes

        self.data = self.data.drop(columns=STR_COL)
        print(f"DF DROPADO: {self.data.head()}\n")

        ## standard scaler pra normalizar dados para media 0 e desvio padrao 1
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(self.data)
        self.data = DataFrame(
            scaled_values, columns=self.data.columns, index=self.data.index
        )
