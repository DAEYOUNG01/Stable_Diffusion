# 1. 테스트 데이터셋 로드 및 전처리
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# CustomDataset 인스턴스 생성
test_dataset = CustomDataset(test_data_list, transform=test_transform)

# 2. DataLoader 생성
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. 모델 로드
num_classes = 6  # bear, cat, deer, dog, eagle, turtle (클래스 수)
model = CustomResNet(num_classes=num_classes)  # CustomResNet은 이미 정의됨
model_path = './checkpoint_resnet18.pt'  # 학습된 모델의 체크포인트 경로

# 4. 모델 평가 함수 정의
def evaluate_model(model, test_loader, device, model_path=None):
    # 모델 가중치 로드
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device)) # strict=False 사용
    model = model.to(device)

    model.eval()

    criterion = nn.CrossEntropyLoss()  # 손실 함수 정의

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():  # 그래디언트 계산 비활성화
        for inputs, labels in tqdm(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)  # 데이터와 라벨을 GPU/CPU로 이동

            outputs = model(inputs)  # 모델 추론
            loss = criterion(outputs, labels)  # 손실 계산
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)  # 예측 결과
            total += labels.size(0)  # 총 샘플 수
            correct += (predicted == labels).sum().item()  # 맞춘 샘플 수

    avg_loss = total_loss / len(test_loader)  # 평균 손실
    accuracy = correct / total * 100  # 정확도 계산
    print(f'Test Accuracy: {accuracy:.2f}%, Avg Loss: {avg_loss:.4f}')

    return accuracy, avg_loss

# 5. 평가 실행 코드
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# evaluate_model 실행
accuracy, avg_loss = evaluate_model(model, test_loader, DEVICE, model_path=model_path)

print(f"Final Test Accuracy: {accuracy:.2f}%")
print(f"Final Test Average Loss: {avg_loss:.4f}")

# 무작위로 데이터 인덱스를 선택
random_index = random.randint(0, len(test_dataset) - 1)

# 선택된 데이터의 이미지와 라벨 가져오기
img, label = test_dataset[random_index]

# 이미지 시각화
plt.imshow(img.permute(1, 2, 0))  # 이미지 차원을 (C, H, W)에서 (H, W, C)로 변환하여 표시
plt.title(f'Label: {test_dataset.classes[label]}')
plt.axis('off')
plt.show()

# 예측 함수 정의 (predict_image 함수는 직접 정의해야 합니다)
def predict_image(img, model):
    model.eval()
    with torch.no_grad():
        img = img.unsqueeze(0)  # 배치 차원을 추가
        img = img.to(DEVICE)    # 모델과 동일한 장치로 이동
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

# 실제 라벨과 예측된 라벨 출력
print('Label:', test_dataset.classes[label], ', Predicted:', test_dataset.classes[predict_image(img, model)])
