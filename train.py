# 7. 손실 함수 및 옵티마이저 정의 (Weight Decay 추가)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # L2 정규화 (Weight Decay)

# 8. 학습률 감소 스케줄러 정의
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 7 에폭마다 학습률 10% 감소
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# 9. 학습 및 평가 함수 정의
def train(model, train_loader, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data, target in train_loader:
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(train_loader), accuracy

# 평가
def evaluate(model, validation_loader):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    """
    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return running_loss / len(validation_loader), accuracy
    """

    with torch.no_grad():
        for data, target in validation_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            loss = criterion(outputs, target)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    accuracy = 100. * correct / total
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=1)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=1)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=1)

    return running_loss / len(validation_loader), accuracy, precision, recall, f1, all_targets, all_preds

# 10. 학습 및 평가
best_train_acc = 0
best_valid_acc = 0 

# 학습 및 평가 과정에서 지표를 저장하고 시각화
train_accuracies = []
valid_accuracies = []
precisions = []
recalls = []
f1_scores = []

for epoch in range(0, 25):
    train_loss, train_accuracy = train(model, train_loader, optimizer)
    valid_loss, valid_accuracy, precision, recall, f1, all_targets, all_preds = evaluate(model, validation_loader)
    #valid_loss, valid_accuracy = evaluate(model, validation_loader)

    train_accuracies.append(train_accuracy)
    valid_accuracies.append(valid_accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)


    """print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.2f}% | "
          f"Valid Loss: {valid_loss:.4f} Valid Acc: {valid_accuracy:.2f}%") """

    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f} Train Acc: {train_accuracy:.2f}% | "
          f"Valid Loss: {valid_loss:.4f} Valid Acc: {valid_accuracy:.2f}% | "
          f"Precision: {precision:.2f} Recall: {recall:.2f} F1 Score: {f1:.2f}")

    # 스케줄러 학습률 감소 적용
    scheduler.step()

    # Best accuracy 갱신
    if valid_accuracy > best_valid_acc:
        best_train_acc, best_valid_acc = train_accuracy, valid_accuracy
        torch.save(model.state_dict(), './checkpoint_resnet18.pt')
