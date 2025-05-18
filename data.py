데이터 증강 및 데이터 셋 구성 

# 2. 데이터 분리 및 전처리
train_data_list = train_55_bear + train_55_cat + train_55_deer + train_55_dog + train_55_eagle + train_55_turtle
#valid_data_list = valid_f_bear + valid_f_cat + valid_f_dear + valid_f_dog + valid_f_eagle + valid_f_turtle

train_label_list = ['55_bear'] * len(train_55_bear) + ['55_cat'] * len(train_55_cat) + ['55_deer'] * len(train_55_deer) + \
                   ['55_dog'] * len(train_55_dog) + ['55_eagle'] * len(train_55_eagle) + ['55_turtle'] * len(train_55_turtle)
"""
valid_label_list = ['f_bear'] * len(valid_f_bear) + ['f_cat'] * len(valid_f_cat) + ['f_dear'] * len(valid_f_dear) + \
                   ['f_dog'] * len(valid_f_dog) + ['f_eagle'] * len(valid_f_eagle) + ['f_turtle'] * len(valid_f_turtle)
"""

classes = ["55_bear", "55_cat", "55_deer", "55_dog", "55_eagle", "55_turtle"]

# 3. 데이터셋 클래스 정의 (PIL 변환 추가)
class CustomDataset(Dataset):
    def __init__(self, data_list, label_list, classes, transform=None):
        self.data_list = data_list
        self.label_list = label_list
        self.classes = classes
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = cv2.imread(img_path)
        
        # 이미지 로드 실패 확인
        if img is None:
            raise ValueError(f"Error: Could not load image at {img_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # OpenCV의 BGR 포맷을 RGB로 변환
        img = Image.fromarray(img)  # numpy 배열을 PIL 이미지로 변환

        if self.transform:
            img = self.transform(img)

        label_name = self.label_list[idx]
        
        # 클래스가 유효한지 확인
        if label_name not in self.classes:
            raise ValueError(f"Error: Label {label_name} not in classes list {self.classes}")
        
        label = self.classes.index(label_name)
        return img, label

# 4. 데이터 변환 (데이터 증강 포함)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # 좌우 반전 추가
    transforms.RandomRotation(10),      # 랜덤 회전 추가
    transforms.Resize((224, 224)),      # 이미지 크기 통일
    transforms.ColorJitter(brightness=0.8, contrast=1.2, saturation=1.5, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. DataLoader 생성
train_dataset = CustomDataset(train_data_list, train_label_list, classes, transform=transform)
valid_dataset = CustomDataset(train_data_list, train_label_list, classes, transform=transform)

train_dataset, valid_dataset = train_test_split(train_dataset, test_size=0.2, random_state=123)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
