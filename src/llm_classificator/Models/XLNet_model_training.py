from transformers import XLNetForSequenceClassification, AdamW
from typing import Any, Dict
from datetime import datetime
from tqdm import tqdm
import torch
from Models import utils


def train_epoch(model, dataloader, optimizer, device):
    model = model.train()
    total_loss = 0
    correct_predictions = 0

    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions += torch.sum(preds == labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss = total_loss / len(dataloader)
    train_accuracy = correct_predictions.double() / len(dataloader.dataset)

    return train_loss, train_accuracy


def xlNet_model_training(data: Dict[str, Any], epochs: int):

    run_name = "XLNet_model"
    estimator_name = "XLNet"
    hyperparams = {"epochs": epochs, "optimizer": "AdamW"}
    training_metrics = {}
    validation_metrics = {}
    lr = 1e-5

    train_dataloader = data["train_dataloader"]
    eval_dataloader = data["eval_dataloader"]

    model_name = "xlnet-base-cased"
    model = XLNetForSequenceClassification.from_pretrained(model_name, num_labels=2)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    print("Execution init datetime: " + str(datetime.now()))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(epochs):
        train_loss, train_accuracy = train_epoch(
            model, train_dataloader, optimizer, device
        )
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss}")

    eval_loss, eval_accuracy = utils.eval_model(model, eval_dataloader, device)

    training_metrics = {"train_loss": train_loss, "train_accuracy": train_accuracy}
    validation_metrics = {"eval_loss": eval_loss, "eval_accuracy": eval_accuracy}

    last_run = utils.track_run(
        run_name, estimator_name, hyperparams, training_metrics, validation_metrics, lr
    )

    # return model
