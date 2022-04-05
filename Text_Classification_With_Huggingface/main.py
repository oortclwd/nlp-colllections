import argparse
import random

from sklearn.matrics import accuracy_score
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizerFast,
    AutoModelForSequenceClassification
)

from .utils import read_text, check_int
from .dataset import TextClassificationDataset, TextClassificationCollator

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True)
    p.add_argument('--train_fn', required=True)
    p.add_argument('--pretrained_model_name', type=str, default='klue/bert-base')

    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=32)
    p.add_argument('--n_epochs', type=int, default=5)

    p.add_argument('--warmup_ratio', type=float, default=.2)

    p.add_argument('--max_length', type=int, default=100)

    config = p.parse_args()

    return config

def get_datasets(fn, valid_ratio=.2):
    labels, texts = read_text(fn)

    if not check_int(label[0]):
        unique_labels = list(set(labels))
        label_to_index = {}
        index_to_label = {}
        for i, label in enumerate(unique_labels):
            label_to_index[label] = i
            index_to_label[i] = label
        labels = [label_to_index[label] for label in labels]
    else:
        unique_labels = sorted(list(set(labels)))
        label_to_index = {}
        index_to_label = {}
        for i, label in enumerate(unique_labels):
            label_to_index[label] = i
            index_to_label[i] = label
        labels = [int(label) for label in labels]

    shuffled = list(zip(texts, labels))
    random.shuffle(shuffled)
    texts = [e[0] for e in shuffled]
    labels = [e[1] for e in shuffled]
    idx = int(len(texts) * (1 - valid_ratio))

    train_dataset = TextClassificationDataset(texts[:idx], labels[:idx])
    valid_dataset = TextClassificationDataset(texts[idx:], labels[idx:])

    return train_dataset, valid_dataset, index_to_label

def main(config):
    tokenizer = AutoTokenizerFast.from_pretrained(config.pretrained_model_name)

    train_dataset, valid_dataset, index_to_label = get_datasets(
        config.train_fn,
        valid_ratio=config.valid_ratio
    )

    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(valid_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config.pretrained_model_name,
        num_labels=len(index_to_label)
    )

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        return {
            'accuracy': accuracy_score(labels, preds)
        }

    training_args = TrainingArguments(
        output_dir='./.checkpoints',
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        weight_decay=0.01,
        fp16=True,
        evaluation_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_steps=n_total_iterations // config.n_epochs,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=TextClassificationCollator(tokenizer,
                                       config.max_length,
                                       with_text=False),
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    torch.save({
        'bert': trainer.model.state_dict(),
        'config': config,
        'classes': index_to_label,
        'tokenizer': tokenizer,
    }, config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
