
# 학습 과정 시각화
fig, axs = plt.subplots(3, 2, figsize=(15, 15))

axs[0, 0].plot(train_accuracies, label='Train Accuracy', color='b')
axs[0, 0].set_title('Train Accuracy')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Accuracy')

axs[0, 1].plot(valid_accuracies, label='Validation Accuracy', color='orange')
axs[0, 1].set_title('Validation Accuracy')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Accuracy')

axs[1, 0].plot(precisions, label='Precision', color='g')
axs[1, 0].set_title('Precision')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Score')

axs[1, 1].plot(recalls, label='Recall', color='r')
axs[1, 1].set_title('Recall')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Score')

axs[2, 0].plot(f1_scores, label='F1 Score', color='purple')
axs[2, 0].set_title('F1 Score')
axs[2, 0].set_xlabel('Epoch')
axs[2, 0].set_ylabel('Score')

fig.tight_layout()
plt.show()

# 오차행렬 시각화
cm = confusion_matrix(all_targets, all_preds)
cmd = ConfusionMatrixDisplay(cm, display_labels=classes)
cmd.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
