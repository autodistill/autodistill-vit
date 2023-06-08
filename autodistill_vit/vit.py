import numpy as np
import supervision as sv
import torch
from autodistill.detection import DetectionTargetModel
from datasets import load_metric
from PIL import Image
from transformers import (Trainer, TrainingArguments, ViTFeatureExtractor,
                          ViTForImageClassification)

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
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def process_image(image_path: str, label: str) -> dict:
    image = Image.open(image_path)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs["labels"] = torch.tensor([label])
    return inputs


def transform(example_batch):
    inputs = feature_extractor([x for x in example_batch["image"]], return_tensors="pt")
    inputs["labels"] = example_batch["labels"]
    return inputs


class ViT(DetectionTargetModel):
    def __init__(self):
        self.vit = None

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        if self.vit is None:
            raise Exception("Model not trained yet!")

        results = self.vit(
            process_image(input, "0")["pixel_values"].to(device),
        )

        scores = results["logits"].tolist()
        labels = results["labels"].tolist()

        detections = sv.Detections(
            xyxy=np.array([]),
            class_id=np.array(labels),
            confidence=np.array(scores),
        )

        return detections

    def train(self, dataset_yaml, epochs=300, image_size=640):
        dataset = sv.detections.core.ClassificationDataset().from_classification_folder(
            dataset_yaml
        )
        labels = dataset.labels()

        dataset_files = dataset.images.keys()
        train_dataset = [
            process_image(f, labels.index(dataset.images[f]))
            for f in dataset_files
            if f.startswith("train")
        ]
        validation_dataset = [
            process_image(f, labels.index(dataset.images[f]))
            for f in dataset_files
            if f.startswith("validation")
        ]

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
            num_train_epochs=4,
            fp16=True,
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
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=feature_extractor,
        )

        trainer.train()

        trainer.save_model("./model")

        self.vit = ViTForImageClassification.from_pretrained("./model")
