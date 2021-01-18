class NAIPSuperResDS(Dataset):
    """
    feature: low resolution (1,13,13)
    target: high resolution tiles (1,390,390)
    """

    def __init__(self, dataset_dir, image_size=390, transform=False):
        """
        Args:
            dataset_dir(string): directory with images
            transform : a torch transformation or false
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.__dir_list = os.listdir(dataset_dir)
        self.transform = transform

    def __len__(self):
        return len(self.__dir_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.__dir_list[idx]

        filename = os.path.join(self.dataset_dir, filename)

        image = torch.tensor(gdal.Open(filename).ReadAsArray().astype(np.float32)) / 255
        image = image.unsqueeze(0)
        # (1,390,390)

        if self.transform:
            image = self.transform(image)

        pool = nn.MaxPool2d(30, stride=30)
        low_res = pool(image)

        sample = {'low_res': low_res, 'high_res': image}

        return sample