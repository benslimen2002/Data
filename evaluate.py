import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import numpy as np
from tqdm import tqdm

def evaluate_model(model, test_loader, device):
    model.eval()

    y_true = []
    y_pred = []
    y_score = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Évaluation"):
            # batch selon ta fonction collate_fn (test_collate_fn)
            q_texts, encodings, imgs, labels, i_labels, img_paths, bodyposes = batch

            encodings = encodings.to(device)
            imgs = imgs.to(device)
            bodyposes = bodyposes.to(device)
            i_labels = i_labels.to(device)

            # === Prédictions ===
            outputs = model(encodings, imgs, bodyposes)  # Ajuste selon ton modèle
            probs = torch.softmax(outputs, dim=1)[:, 1]   # proba de la classe positive
            preds = torch.argmax(outputs, dim=1)

            # Stocker les résultats
            y_true.extend(i_labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_score.extend(probs.cpu().numpy())

    # Conversion en numpy
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_score = np.array(y_score)

    # === Calcul métriques ===
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_score)

    print("\n===== Résultats Évaluation =====")
    print(f"Précision : {precision:.4f}")
    print(f"Rappel    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"AUC       : {auc:.4f}")
    print("\nRapport de classification :\n")
    print(classification_report(y_true, y_pred, digits=4))

    return precision, recall, f1, auc
