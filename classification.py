from dataset import *

class Classification:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.X = dataset.X[:, 1:]
        self.y = dataset.X[:, 0]

        self.attributeNames = dataset.attributeNames
        self.attributeNames.remove('Area')

if __name__ == '__main__':
    regression = Classification(Dataset(uci_id = 545))
    print(regression.attributeNames)
    print(regression.X)
