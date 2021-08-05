import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pycm import ConfusionMatrix
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from tokenizers import Tokenizer
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.utils import (
    filter_labels,
    filter_paths,
    get_cod,
    get_summary_paths,
    read_file,
    read_label,
    tokenize_to_vec,
)
from src.weighting import tf_idf, z_norm

# Prepare file paths
CWD = Path.cwd()
DATA = Path("/work/fabiasch/eia/data")
SEG = DATA.joinpath("filtered2")

# Read files
# read labels into Tuple[Dict[COD-str, 0 or 1], np.array]
labels, y = read_label("./label_extra_pos.csv")
# get all pathlib.Paths to legislative proposal summaries
file_paths = get_summary_paths(SEG, final_act=False)
# split Paths into (label, other) paths
train_paths, other_paths = filter_labels(file_paths, labels)
# load pre-trained huggingface tokenizer
tokenizer = Tokenizer.from_file("./tokenizer.lr.json")

# Prepare training data
# read sentence-splitted legislative proposal summaries into List[List[str]]
train_sum = [read_file(x) for x in train_paths]
other_sum = [read_file(x) for x in other_paths]
all_sum = train_sum + other_sum

X_train, X_test, y_train, y_test = train_test_split(
    train_sum, y, test_size=0.2, random_state=42, stratify=y
)


class PlainDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx])


class Model(LightningModule):
    def __init__(self, model_name_or_path: str, lr=2e-5):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.lr = lr
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.hparams.model_name_or_path, num_labels=2
        )

    def forward(self, batch):
        return self.model(**batch).logits

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        outputs = self.model(**batch)
        self.log("val_loss", outputs.loss)
        return (outputs.logits, batch["labels"])

    def validation_epoch_end(self, validation_step_outputs):
        logits, labels = zip(*validation_step_outputs)
        preds = torch.vstack(logits).argmax(1)
        labels = torch.cat(labels).flatten()
        val_acc = (preds == labels).sum() / labels.shape[0]
        self.log("val_acc", val_acc)
        print(f"{val_acc=}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, inputs):
        documents, labels = zip(*inputs)
        documents = [x[0] for x in documents]
        batch = self.tokenizer(
            documents, return_tensors="pt", padding=True, truncation=True
        )
        batch["labels"] = torch.LongTensor(labels)
        return batch


checkpoint_path = "/work/fabiasch/eia/legis_sum_coding/lightning_logs/version_19/checkpoints/epoch=3-step=55.ckpt"
model_name_or_path = "distilroberta-base"
batch_size = 6
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
collate_fn = Collator(tokenizer)
train_dataset = PlainDataset(data=X_train, labels=y_train)
val_dataset = PlainDataset(data=X_test, labels=y_test)
train_loader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn,
)
trainer = Trainer(
    gpus="0", max_epochs=15, callbacks=[ModelCheckpoint(monitor="val_acc", mode="max")]
)

if checkpoint_path is None:
    model = Model(model_name_or_path=model_name_or_path)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)

model = Model.load_from_checkpoint(checkpoint_path=checkpoint_path)
logits = np.vstack(trainer.predict(model, dataloaders=val_loader))
probs = F.softmax(torch.from_numpy(logits), -1)
preds = probs.argmax(1)
print((preds.numpy() == y_test).sum() / y_test.shape[0])

other_dataset = PlainDataset(
    data=other_sum, labels=torch.ones((len(other_sum),)).long()
)
other_loader = DataLoader(
    other_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn,
)
other_logits = np.vstack(trainer.predict(model, dataloaders=other_loader))
other_probs = F.softmax(torch.from_numpy(other_logits), -1)
other_preds = other_probs.argmax(1)
# print((preds.numpy() == y_test).sum() / y_test.shape[0])

cod = [get_cod(p) for p in other_paths]
out = {
    "COD": cod,
    "Transformer_Predictions": other_preds.tolist(),
    "Transformer_Probabilities": other_probs.tolist(),
}
df = pd.DataFrame.from_dict(out)
df.to_csv("transformer_output.csv")


final_act_paths = get_summary_paths(SEG, final_act=True)
final_act_sum = [read_file(x) for x in final_act_paths]

legis_sum_dataset = PlainDataset(
    data=all_sum, labels=torch.ones((len(all_sum),)).long()
)
legis_sum_loader = DataLoader(
    legis_sum_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn,
)
legis_sum_logits = np.vstack(trainer.predict(model, dataloaders=legis_sum_loader))
legis_sum_probs = F.softmax(torch.from_numpy(legis_sum_logits), -1)
legis_sum_preds = legis_sum_probs.argmax(1)

final_act_dataset = PlainDataset(
    data=final_act_sum, labels=torch.ones((len(final_act_sum),)).long()
)
final_act_loader = DataLoader(
    final_act_dataset,
    shuffle=False,
    batch_size=batch_size,
    num_workers=4,
    collate_fn=collate_fn,
)
final_act_logits = np.vstack(trainer.predict(model, dataloaders=final_act_loader))
final_act_probs = F.softmax(torch.from_numpy(final_act_logits), -1)
final_act_preds = final_act_probs.argmax(1)

legis_sum_cod = [get_cod(p) for p in train_paths]
final_act_cod = [get_cod(p) for p in final_act_paths]
overlapping_paths = list(set(legis_sum_cod).intersection(set(final_act_cod)))

filtered_legis_sum_probs = filter_paths(legis_sum_probs, train_paths, overlapping_paths)
filtered_legis_sum_preds = filter_paths(legis_sum_preds, train_paths, overlapping_paths)

filtered_final_act_probs = filter_paths(final_act_probs, train_paths, overlapping_paths)
filtered_final_act_preds = filter_paths(final_act_preds, train_paths, overlapping_paths)

filtered_legis_sum_probs = np.around(
    torch.vstack(filtered_legis_sum_probs)[:, 1].numpy(), 2
)
filtered_final_act_probs = np.around(
    torch.vstack(filtered_final_act_probs)[:, 1].numpy(), 2
)

filtered_legis_sum_preds = torch.vstack(filtered_legis_sum_preds).numpy().flatten()
filtered_final_act_preds = torch.vstack(filtered_final_act_preds).numpy().flatten()

df_dico = {
    "COD": overlapping_paths,
    "Proposal_Transformer_LR_Prediction": filtered_legis_sum_preds,
    "Proposal_Transformer_LR_Probability": filtered_legis_sum_probs,
    "Final_Act_Transformer_LR_Prediction": filtered_final_act_preds,
    "Final_Act_Transformer_LR_Probability": filtered_final_act_probs,
    "Transformer_Same_Class": filtered_legis_sum_preds == filtered_final_act_preds,
    "Transformer_Prob_dif": np.around(
        np.array(filtered_final_act_probs) - np.array(filtered_legis_sum_probs), 2
    ),
}
df = pd.DataFrame.from_dict(df_dico)
df.to_csv("210531_transformer.full-preds.csv", index=False)
