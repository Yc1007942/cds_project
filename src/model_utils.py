import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

def calculate_r2_score(y_true, y_pred):
    return r2_score(y_true, y_pred)

def train(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_preds = []
    total_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        input_ids = batch["input_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        total_preds.extend(preds.detach().cpu().numpy())
        total_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(total_labels, total_preds)
    f1score = f1_score(total_labels, total_preds)

    return avg_loss, accuracy, f1score


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_preds = []
    total_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_masks = batch["attention_masks"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            total_preds.extend(preds.detach().cpu().numpy())
            total_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(total_labels, total_preds)
    f1score = f1_score(total_labels, total_preds)

    return avg_loss, accuracy, f1score


def training_loop(model, train_loader, val_loader, optimizer, early_stopping, num_epochs, device):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_f1s, val_f1s = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss, train_acc, train_f1 = train(
            model, train_loader, optimizer, device
        )

        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)

        early_stopping(val_acc, model)

        if early_stopping.early_stop:
            print("Early stopping triggered. Halting training.")
            break

    print(f"\nTraining Complete. Loading best weights from '{early_stopping.path}'...")
    model.load_state_dict(torch.load(early_stopping.path, weights_only=True))

    return train_losses, val_losses, train_accs, val_accs, train_f1s, val_f1s

def evaluate_on_testing_set(y_test, y_pred):
    print("AUC is: ", roc_auc_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix: \n", confusion_matrix(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc_score(y_test, y_pred))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
