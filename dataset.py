# CustomDataset 클래스 정의

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform
        self.classes = ["bear", "cat", "deer", "dog", "eagle", "turtle"]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        # 이미지 경로에서 클래스명 추출
        label_name = os.path.basename(os.path.dirname(img_path))
        label = self.classes.index(label_name)

        return img, label
