import os
import json
import torch
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, f1_score

from modules.Ped_dataset import PedDataset
from modules.Ped_model import PedVLMT5
from transformers import T5Tokenizer, GPT2TokenizerFast, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def params():
    parser = argparse.ArgumentParser()

    # chemins
    parser.add_argument('--optical', type=bool, default=False)
    parser.add_argument('--img_path', default='',  help='Path to labels')
    parser.add_argument("--checkpoint-file", type=str, required=True, help="Checkpoint à charger")
    parser.add_argument("--data_path", default=r"C:\Users\DELL\Downloads\Ped_VLM.-main\Ped_Dataset", help="Dataset path")
    parser.add_argument("--result_path", default=r"C:\Users\DELL\Downloads\Ped_VLM.-main\results", help="Results path")
    parser.add_argument("--bodypose_path", default=r"C:\Users\DELL\Downloads\Ped_VLM.-main\data\JAAD\bodypose_yolo", help="Bodypose path")

    # dataset
    parser.add_argument("--batch-size", default=1, type=int, help="Batch size for evaluation")
    parser.add_argument("--num-workers", default=0, type=int, help="Dataloader workers")
    parser.add_argument("--use-bodypose", default=True, type=bool, help="Use bodypose features")

    # modèle
    parser.add_argument("--lm", default="T5-Base", type=str, choices=["T5-Base", "T5-Small", "T5-Large"], help="Language model backbone")
    parser.add_argument("--encoder", default="clip", type=str, choices=["clip", "vit", "resnet"], help="Image encoder backbone")
    parser.add_argument("--freeze_lm", action="store_true", help="Freeze LM parameters")
    parser.add_argument("--lora", action="store_true", help="Use LoRA adaptation")
    parser.add_argument("--gpa_hidden_size", default=128, type=int, help="Hidden size for GPA module")
    parser.add_argument("--num_head", default=8, type=int, help="Number of attention heads")
    parser.add_argument('--attention', default=False, type=bool, help='Fuse feature using attention')


    args = parser.parse_args()
    return args






def evaluate(model, dataloader, criterion):
    model.eval()
    y_true, y_pred, y_scores = [], [], []
    total_loss = 0

    with torch.no_grad():
        for inputs, imgs, labels, i_labels, bodyposes in tqdm(dataloader, desc="Evaluating"):
            inputs, imgs, labels, i_labels, bodyposes = (
                inputs.to(device),
                imgs.to(device),
                labels.to(device),
                i_labels.to(device),
                bodyposes.to(device),
            )

            outputs, out_int = model(inputs, imgs, labels, bodyposes)
            loss = criterion(out_int, i_labels)
            total_loss += loss.item()

            preds = torch.argmax(out_int, dim=1)
            probs = torch.softmax(out_int, dim=1)[:, 1]

            y_true.extend(i_labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except:
        auc = 0.0
    cm = confusion_matrix(y_true, y_pred)

    return {
        "loss": total_loss / len(dataloader),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
    }

if __name__ == "__main__":
    config = params()

    # --- Tokenizer ---
    if config.lm == "T5-Base":
        processor = T5Tokenizer.from_pretrained("google-t5/t5-base", model_max_length=1024)
        processor.add_tokens("<")
    elif config.lm == "GPT":
        processor = GPT2TokenizerFast.from_pretrained("gpt2")
        processor.pad_token = processor.eos_token
    elif config.lm == "PN":
        processor = AutoTokenizer.from_pretrained("google/pegasus-xsum")
        processor.add_tokens(["<"])
    else:
        processor = T5Tokenizer.from_pretrained("google-t5/t5-large")
        processor.add_tokens("<")

    # --- Dataset ---
    test_dset = PedDataset(
        input_file=os.path.join(config.data_path, "test.json"),
        config=config,
        tokenizer=processor,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5)),
        ]),
    )
    test_dset.use_bodypose = config.use_bodypose

    test_dataloader = DataLoader(
        test_dset,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        collate_fn=test_dset.collate_fn,
    )

    # --- Charger modèle ---
    model = PedVLMT5(config)
    if os.path.isfile(config.checkpoint_file):
     checkpoint_path = config.checkpoint_file  # chemin complet fourni
    else:
     checkpoint_path = os.path.join(config.result_path, config.checkpoint_file, 'latest_model.pth')
   
    model = PedVLMT5(config)  # crée l'instance du modèle
    state_dict = torch.load(checkpoint_path, map_location=device)  # charge uniquement les poids
    model.load_state_dict(state_dict, strict=False) # applique les poids au modèle
    model.to(device)  # envoie le modèle sur GPU/CPU


    criterion = torch.nn.CrossEntropyLoss()
    results = evaluate(model, test_dataloader, criterion)

    print(json.dumps(results, indent=4))
