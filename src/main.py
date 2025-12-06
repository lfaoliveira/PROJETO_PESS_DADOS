from torch.utils.data import DataLoader

from DataProcesser.data import StrokeDataset

## AUX_VARS
BATCH_SIZE = 8
WORKERS = 1
shuffle = False

dataset = StrokeDataset()

dataloader: DataLoader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=WORKERS)


if __name__ == "__main__":
    for row, label in dataloader:
        print(row, label)
