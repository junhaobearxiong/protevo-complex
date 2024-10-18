from torch.utils.data import Dataset, DataLoader


class PairMSADataset(Dataset):
    def __init__(self, pair_name: str, transitions_dir: str):
        """
        Args:
        pair_name (str): "{protein_1}_{protein_2}"
        transitions_dir(str): Directory storing all transitions, each file is
            "{protein_x}_{protein_y}.txt"
        """
        super().__init__()

    def __len__(self):
        pass

    def __getitem__(self, index):
        return super().__getitem__(index)
