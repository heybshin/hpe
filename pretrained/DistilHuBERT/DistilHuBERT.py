from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, EvalPrediction
from datasets import Audio, Dataset, DatasetDict
import numpy as np
from datasets import load_dataset
import evaluate
import pandas as pd
import os
import re
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import torch


# source: https://jesusleal.io/2021/04/21/Longformer-multilabel-classification/
def multi_label_metrics(predictions, labels, threshold=0.5):
    # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    # next, use threshold to turn them into integer predictions
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    # finally, compute metrics
    y_true = labels
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    roc_auc = roc_auc_score(y_true, y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    # return as dictionary
    metrics = {'f1': f1_micro_average,
               'roc_auc': roc_auc,
               'accuracy': accuracy}
    return metrics


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(predictions=preds, labels=p.label_ids)
    return result
# def compute_metrics(eval_pred):
#     """Computes accuracy on a batch of predictions"""
#     predictions = np.argmax(eval_pred.predictions, axis=1)
#     return metric.compute(predictions=predictions, references=eval_pred.label_ids)


def preprocess_function(examples):

    max_duration = 1.0

    audio_arrays = [x["array"] for x in examples["audio"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        truncation=True,
        return_attention_mask=True,
    )
    labels_batch = {k: examples[k] for k in examples.keys() if k in labels}
    labels_matrix = np.zeros((len(audio_arrays), len(labels)))
    for idx, label in enumerate(labels):
        labels_matrix[:, idx] = labels_batch[label]
    inputs["labels"] = labels_matrix.tolist()

    return inputs



def parse_labels(filename):
    pattern = r"note(\d+)_velocity(\d+)"
    match = re.search(pattern, filename)
    if match:
        note = int(match.group(1))
        velocity = int(match.group(2))
        return note, velocity
    return None, None


def build_distilhubert(ckpt_path):

    dataset = DatasetDict.load_from_disk("data/filtered_dataset")

    labels = [label for label in dataset['train'].features.keys() if label not in ['audio', 'label']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    num_labels = len(id2label)

    # model_id = "ntu-spml/distilhubert"
    model = AutoModelForAudioClassification.from_pretrained(
        ckpt_path,
        problem_type="multi_label_classification",
        num_labels=num_labels,  # for audio
        label2id=label2id,
        id2label=id2label,
    )
    model.projector = torch.nn.Linear(768, 368)
    model.classifier = torch.nn.Linear(368, num_labels)

    return model


if __name__ == "__main__":

    # # first create more columns in the column_names as the labels, the new created column name is velocity1, velocity2, ... velocity10 and note1, note1, note2, ... note127
    # # create labels for each audio file, based on the value of velocity and note in the file name
    # # e.g. ClassicElectricPiano_note0_velocity1.wav -> note0, velocity1 True and others False
    # # Filter out the data with all zeros at the beginning
    #
    # dataset = load_dataset("audiofolder", data_dir="../data/audios")
    # dataset = dataset.filter(lambda x: not is_all_zero(x["audio"]["array"]))
    # model_id = "ntu-spml/distilhubert"
    #
    # feature_extractor = AutoFeatureExtractor.from_pretrained(
    #     model_id, do_normalize=True, return_attention_mask=True
    # )
    # sampling_rate = feature_extractor.sampling_rate
    #
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    #
    # # Create a DataFrame to hold the additional labels
    # df = pd.DataFrame(dataset['train'])
    #
    # # Initialize the new columns with False
    # note_columns = {f"note{i}": False for i in range(128)}
    # velocity_columns = {f"velocity{i}": False for i in range(1, 11)}
    # timbre_columns = {f"timbre{i}": False for i in range(0, 16)}
    # df = df.assign(**note_columns, **velocity_columns, **timbre_columns)
    #
    # # Set the appropriate labels to True based on the filename
    # for idx, row in df.iterrows():
    #     filename = os.path.basename(row['audio']['path'])
    #     note, velocity = parse_labels(filename)
    #     if note is not None and velocity is not None:
    #         df.at[idx, f"note{note}"] = True
    #         df.at[idx, f"velocity{velocity}"] = True
    #         df.at[idx, f"timbre{df.at[idx, 'label']}"] = True
    #         # get the label of that file and assign it to the column
    #
    # # Merge the new columns back into the dataset
    # # dataset['train'] = Dataset.from_pandas(df)
    # dataset['train'] = Dataset.from_dict(df)
    # # filtered_dataset = dataset.filter(lambda x: not is_all_zero(x["audio"]["array"]))
    # dataset.save_to_disk("../data/filtered_dataset")

    dataset = DatasetDict.load_from_disk("../../data/filtered_dataset")

    # Split the filtered dataset into train and test sets
    dataset = dataset["train"].train_test_split(seed=42, shuffle=True, test_size=0.1)

    # id2label_fn = dataset["train"].features["label"].int2str
    labels = [label for label in dataset['train'].features.keys() if label not in ['audio', 'label']]
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}

    model_id = "ntu-spml/distilhubert"

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_id, do_normalize=True, return_attention_mask=True
    )

    # sampling_rate = feature_extractor.sampling_rate
    #
    # dataset = dataset.cast_column("audio", Audio(sampling_rate=sampling_rate))
    sample = dataset["train"][0]["audio"]
    print(f"Mean: {np.mean(sample['array']):.3}, Variance: {np.var(sample['array']):.3}")

    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])

    print(f"inputs keys: {list(inputs.keys())}")
    print(f"Mean: {np.mean(inputs['input_values']):.3}, Variance: {np.var(inputs['input_values']):.3}")

    # dataset_encoded = dataset.map(
    #     preprocess_function,
    #     remove_columns=["audio"],
    #     batched=True,
    #     batch_size=100,
    #     num_proc=1,
    # )
    dataset_encoded = dataset.map(
        preprocess_function,
        remove_columns=dataset['train'].column_names,
        batched=True,
        batch_size=100,
        num_proc=1,
    )
    dataset_encoded.set_format("torch")
    # encoded_dataset = dataset.map(preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

    # dataset_encoded = dataset_encoded.rename_column("genre", "label")
    # id2label = {str(i): id2label_fn(i) for i in range(len(dataset_encoded["train"].features["label"].names))}
    # label2id = {v: k for k, v in id2label.items()}

    # Fine-tuning the model
    num_labels = len(id2label)
    model = AutoModelForAudioClassification.from_pretrained(
        model_id,
        problem_type="multi_label_classification",
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )
    model.projector = torch.nn.Linear(768, 368)
    model.classifier = torch.nn.Linear(368, num_labels)

    model_name = model_id.split("/")[-1]
    batch_size = 8
    gradient_accumulation_steps = 1
    num_train_epochs = 100
    metric_name = "f1"

    training_args = TrainingArguments(
        f"{model_name}-finetuned-dataset",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        warmup_ratio=0.01,
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        fp16=True,
        # push_to_hub=True,
    )

    # metric = evaluate.load("accuracy")

    trainer = Trainer(
        model,
        training_args,
        # train_dataset=dataset_encoded["train"].with_format("torch"),
        # eval_dataset=dataset_encoded["test"].with_format("torch"),
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    # trainer.evaluate()
    trainer.save_model("../outputs/DistilHuBERT")
    # example = dataset_encoded['train'][0]
    # example['labels']
    # dataset_encoded['train'][0]['labels'].type()
    # dataset_encoded['train']['input_ids'][0]

    # input_data = dataset_encoded['train']['input_values'][0].unsqueeze(0).cuda()
    # labels = dataset_encoded['train']['labels'][0].unsqueeze(0).cuda()
    # model = model.cuda()
    # outputs = model(input_values=input_data, labels=labels)
    # outputs = model(input_values=dataset_encoded['train']['input_values'][0].unsqueeze(0), labels=dataset_encoded['train'][0]['labels'].unsqueeze(0))