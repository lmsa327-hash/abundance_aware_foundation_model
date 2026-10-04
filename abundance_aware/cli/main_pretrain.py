import torch
from torch.utils.data import random_split
import pandas as pd
from pickle import load, dump

import os
import warnings
from transformers import (
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, GPT2LMHeadModel,
)
from transformers.trainer_callback import EarlyStoppingCallback

from abundance_aware.src.Datasets.MicrobialCorpus import MicrobialCorpusWithLabelTokens
from abundance_aware.src.MultiEmbeddings.models.MultiEmbeddingsDataCollator import MultiEmbeddingsDataCollator
from abundance_aware.src.MultiEmbeddings.models.MultiEmbeddingsModel import MultiEmbeddingsGPT2LMHeadModel
from abundance_aware.src.MultiEmbeddings.utils import build_conditioners_from_config
from abundance_aware.src.utils.callbacks import LambdaATrackingCallback, ConditionerTrackingCallback

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")  # For tensor cores
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import mlflow
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("abundance-attn-pretrain")

def pretrain(cfg, args):
    corpus = load(open(args.input, 'rb'))
    tokenizer = corpus.tokenizer

    config = {
        'model_type': cfg.get('GPT2', 'model_type'),
        'vocab_size': tokenizer.vocab_size,
        'n_positions': cfg.getint('GPT2', 'n_positions'),
        'n_embd': cfg.getint('GPT2', 'n_embd'),
        'n_layer': cfg.getint('GPT2', 'n_layer'),
        'n_head': cfg.getint('GPT2', 'n_head'),
        'bos_token_id': tokenizer.bos_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
    config = GPT2Config(**config)

    conditioners_spec = build_conditioners_from_config(cfg)

    training_args = TrainingArguments(**{
        "learning_rate": cfg.getfloat('pretrain', 'learning_rate'),
        "do_train": True,
        "do_eval": True,
        "train_sampling_strategy": "group_by_length",
        "length_column_name": "length",
        "disable_tqdm": False,
        "lr_scheduler_type": "linear",
        "warmup_steps": cfg.getint('pretrain', 'warmup_steps'),
        "weight_decay": cfg.getfloat('pretrain', 'weight_decay'),
        "per_device_train_batch_size": cfg.getint('pretrain', 'per_device_train_batch_size'),
        "num_train_epochs": cfg.getint('pretrain', 'num_train_epochs'),
        "eval_strategy": "steps",
        "eval_steps": cfg.getint('pretrain', 'eval_steps'),
        "save_strategy": "steps",
        "save_steps": cfg.getint('pretrain', 'save_steps'),
        "logging_steps": cfg.getint('pretrain', 'logging_steps'),
        "output_dir": f'{args.log}/pretrain_checkpoints',
        "load_best_model_at_end": True,
        "report_to": ["mlflow"],
    })

    if args.with_label:
        if args.labels is None:
            raise ValueError("Please provide labels for pretraining.")
        metadata = pd.read_csv(args.labels, index_col=0)
        metadata = metadata.loc[corpus.data.index]
        extend_words = metadata.iloc[:, 0].unique().tolist()
        tokenizer.add_tokens(extend_words)
        os.makedirs(args.output, exist_ok=True)
        dump(tokenizer, open(f'{args.output}/tokenizer.pkl', 'wb'))
        corpus = MicrobialCorpusWithLabelTokens(
            corpus.tokens, corpus.abundances, metadata.iloc[:, 0].values.tolist(), tokenizer
        )

    print(f"Start training...")
    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.from_scratch:
        model = MultiEmbeddingsGPT2LMHeadModel(config, conditioners=conditioners_spec).to(device)
        print("Training from scratch.")
    else:
        model = MultiEmbeddingsGPT2LMHeadModel.from_pretrained(
            args.model, conditioners=conditioners_spec
        ).to(device)

    if args.with_label:
        model.resize_token_embeddings(len(tokenizer))
        print("Update the embedding layer to include the label embedding.")

    data_collator = MultiEmbeddingsDataCollator(base_collator, model.conditioners, tokenizer.pad_token_id)

    model = model.train()
    train_set, val_set = random_split(corpus, [1 - args.val_split, args.val_split])

    callbacks = [EarlyStoppingCallback(early_stopping_patience=3), ConditionerTrackingCallback()]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    trainer.train()
    os.makedirs(args.output, exist_ok=True)
    trainer.save_model(args.output)

    logs = pd.DataFrame(trainer.state.log_history)
    os.makedirs(args.log, exist_ok=True)
    logs.to_csv(os.path.join(args.log, "pretrain_log.csv"), index=False)