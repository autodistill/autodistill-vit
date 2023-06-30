import numpy as np
import supervision as sv
import torch
from autodistill.detection import DetectionTargetModel
from datasets import Dataset, DatasetDict, Image, load_metric
from transformers import (Trainer, TrainingArguments, ViTFeatureExtractor,
                          ViTForImageClassification)
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

feature_extractor = ViTFeatureExtractor.from_pretrained(
    "google/vit-base-patch16-224-in21k"
)

metric = load_metric("accuracy")

def compute_metrics(p):
    return metric.compute(
        predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
    )


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch], dtype=torch.float),
    }


def transform(example_batch):
    # Take a list of PIL images and turn them to pixel values
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")

    # Don't forget to include the labels!
    inputs["labels"] = example_batch["labels"]
    return inputs


class ViT(DetectionTargetModel):
    def __init__(self):
        self.vit = None

    def predict(self, input: str) -> sv.Classifications:
        if self.vit is None:
            raise Exception("Model not trained yet!")

        image = cv2.imread(input)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        inputs = feature_extractor(images=image, return_tensors="pt")

        outputs = self.vit(**inputs)

        labels = np.argmax(outputs.logits.detach().cpu().numpy(), axis=1)

        scores = np.max(outputs.logits.detach().cpu().numpy(), axis=1)

        return sv.Classifications(
            class_id=labels,
            confidence=scores,
        )

    def train(self, dataset_path, epochs=10) -> None:
        dataset = sv.ClassificationDataset.from_multiclass_folder_structure(dataset_path)

        labels = dataset.classes
        annotations = dataset.annotations

        hf_dataset = Dataset.from_dict(
            {
                "image": [os.path.join(dataset_path, x) for x in annotations.keys()],
                "labels": [x.class_id[0] for x in annotations.values()],
            }
        )

        train = hf_dataset.filter(lambda x: x["image"].split("/")[-3] == "train")
        test = hf_dataset.filter(lambda x: x["image"].split("/")[-3] == "test")

        prepared_ds = DatasetDict({"train": train, "test": test})

        # cast to PIL images
        prepared_ds["train"] = prepared_ds["train"].cast_column("image", Image())

        prepared_ds["test"] = prepared_ds["test"].cast_column("image", Image())

        prepared_ds["train"] = prepared_ds["train"].with_transform(transform)
        prepared_ds["test"] = prepared_ds["test"].with_transform(transform)

        model = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=len(labels),
            id2label={str(i): c for i, c in enumerate(labels)},
            label2id={c: str(i) for i, c in enumerate(labels)},
        )

        training_args = TrainingArguments(
            output_dir="./vit-base-beans",
            per_device_train_batch_size=16,
            evaluation_strategy="steps",
            num_train_epochs=epochs,
            fp16=True if device == "cuda" else False,
            save_steps=100,
            eval_steps=100,
            logging_steps=10,
            learning_rate=2e-4,
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="tensorboard",
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            train_dataset=prepared_ds["train"],
            eval_dataset=prepared_ds["test"],
            tokenizer=feature_extractor,
        )

        trainer.train()

        trainer.save_model("./model")

        self.vit = ViTForImageClassification.from_pretrained("./model")
