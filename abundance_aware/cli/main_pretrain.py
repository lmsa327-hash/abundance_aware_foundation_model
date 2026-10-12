import configparser
import gc
import math
import os
import warnings
from pickle import load, dump

import mlflow
import pandas as pd
import torch
from torch.utils.data import random_split
from transformers import (
    GPT2Config,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from transformers.trainer_callback import EarlyStoppingCallback

from abundance_aware.cli.utils import ABLATIONS
from abundance_aware.src.Datasets.CustomUnPickler import CustomUnpickler
from abundance_aware.src.Datasets.MicrobialCorpus import MicrobialCorpusWithLabelTokens, MicrobialCorpus
from abundance_aware.src.MultiEmbeddings.models.MultiEmbeddingsDataCollator import MultiEmbeddingsDataCollator
from abundance_aware.src.MultiEmbeddings.models.MultiEmbeddingsModel import MultiEmbeddingsGPT2LMHeadModel
from abundance_aware.src.MultiEmbeddings.utils import build_conditioners_from_config
from abundance_aware.src.utils.Utils import find_pkg_resource
from abundance_aware.src.utils.callbacks import ConditionerTrackingCallback

warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision("medium")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tracking_uri = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_registry_uri(tracking_uri)
mlflow.set_experiment("abundance-attn-ablation")  # separate from the main pretrain runs


def apply_ablation(cfg, flags):
    """Return a copy of cfg with [conditioners] and [biases] set for this variant.

    use_abundance / use_hierarchy -> which components are enabled
    use_independent_attention     -> also enable the attention biases for those components
    use_gate                      -> use_gated_fusion on every enabled conditioner / bias
    """
    new = configparser.ConfigParser()
    new.read_dict({s: dict(cfg.items(s, raw=True)) for s in cfg.sections()})

    components = [
        name for name, on in (
            ("abundance", flags["use_abundance"]),
            ("hierarchy", flags["use_hierarchy"]),
        ) if on
    ]
    gate = str(bool(flags["use_gate"]))  # "True" / "False" parse correctly with ast.literal_eval

    def configure(group, prefix, enabled):
        if not new.has_section(group):
            new.add_section(group)
        new.set(group, "enabled", ", ".join(enabled))
        for name in enabled:
            section = f"{prefix}:{name}"
            if not new.has_section(section):
                raise KeyError(f"[{section}] missing from config")
            new.set(section, "use_gated_fusion", gate)

    configure("conditioners", "conditioner", components)
    configure("biases", "bias", components if flags["use_independent_attention"] else [])
    return new


def validate_flags(name, flags):
    """Catch impossible variants early (attention/gating need something to act on)."""
    has_signal = flags["use_abundance"] or flags["use_hierarchy"]
    if (flags["use_independent_attention"] or flags["use_gate"]) and not has_signal:
        raise ValueError(f"{name}: attention/gating enabled without abundance or hierarchy.")

def pretrain(cfg, args, ablation_name):
    flags = ABLATIONS[ablation_name]
    validate_flags(ablation_name, flags)

    # Everything that differs between variants goes under a per-variant folder
    out_dir = os.path.join(args.output, ablation_name, f"seed{args.seed}")
    log_dir = os.path.join(args.log, ablation_name, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    set_seed(args.seed)

    corpus = load(open(args.input, "rb"))
    tokenizer = corpus.tokenizer

    config = GPT2Config(
        model_type=cfg.get("GPT2", "model_type"),
        loss_type="ForCausalLM",
        vocab_size=tokenizer.vocab_size,
        n_positions=cfg.getint("GPT2", "n_positions"),
        n_embd=cfg.getint("GPT2", "n_embd"),
        n_layer=cfg.getint("GPT2", "n_layer"),
        n_head=cfg.getint("GPT2", "n_head"),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    abl_cfg = apply_ablation(cfg, flags)
    conditioners_spec = build_conditioners_from_config(abl_cfg)
    print("conditioners: ",conditioners_spec)
    training_args = TrainingArguments(
        learning_rate=cfg.getfloat("pretrain", "learning_rate"),
        do_train=True,
        do_eval=True,
        train_sampling_strategy="group_by_length",
        length_column_name="length",
        disable_tqdm=False,
        lr_scheduler_type="linear",
        warmup_steps=cfg.getint("pretrain", "warmup_steps"),
        weight_decay=cfg.getfloat("pretrain", "weight_decay"),
        per_device_train_batch_size=cfg.getint("pretrain", "per_device_train_batch_size"),
        num_train_epochs=cfg.getint("pretrain", "num_train_epochs"),
        eval_strategy="steps",
        eval_steps=cfg.getint("pretrain", "eval_steps"),
        save_strategy="steps",
        save_steps=cfg.getint("pretrain", "save_steps"),
        logging_steps=cfg.getint("pretrain", "logging_steps"),
        output_dir=f"{log_dir}/pretrain_checkpoints",
        load_best_model_at_end=True,
        report_to=["mlflow"],
        seed=args.seed,
        data_seed=args.seed,
    )

    if args.with_label:
        if args.labels is None:
            raise ValueError("Please provide labels for pretraining.")
        metadata = pd.read_csv(args.labels, index_col=0)
        metadata = metadata.loc[corpus.data.index]
        extend_words = metadata.iloc[:, 0].unique().tolist()
        tokenizer.add_tokens(extend_words)
        dump(tokenizer, open(f"{out_dir}/tokenizer.pkl", "wb"))
        corpus = MicrobialCorpusWithLabelTokens(
            corpus.tokens, corpus.abundances, metadata.iloc[:, 0].values.tolist(), tokenizer
        )

    base_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.from_scratch:
        model = MultiEmbeddingsGPT2LMHeadModel(config, conditioners=conditioners_spec).to(device)
    else:
        model = MultiEmbeddingsGPT2LMHeadModel.from_pretrained(
            args.model, conditioners=conditioners_spec
        ).to(device)

    if args.with_label:
        model.resize_token_embeddings(len(tokenizer))

    # NOTE: for M0 (no conditioners) make sure this collator handles an empty conditioner set.
    data_collator = MultiEmbeddingsDataCollator(base_collator, model.conditioners, tokenizer.pad_token_id)

    # Same split for every variant (depends only on split_seed, not on the ablation)
    split_gen = torch.Generator().manual_seed(0)
    train_set, val_set = random_split(corpus, [1 - args.val_split, args.val_split], generator=split_gen)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3), ConditionerTrackingCallback()],
    )

    # Training must happen INSIDE the run, otherwise metrics land in a different run
    with mlflow.start_run(run_name=f"{ablation_name}_seed{args.seed}") as run:
        mlflow.set_tags({"ablation": ablation_name, "seed": str(args.seed)})
        mlflow.log_params({
            **{f"abl_{k}": v for k, v in flags.items()},
            "ablation": ablation_name,
            "seed": args.seed,
            "split_seed": 0,
            "from_scratch": args.from_scratch,
            "with_label": args.with_label,
            "val_split": args.val_split,
            "learning_rate": training_args.learning_rate,
            "num_train_epochs": training_args.num_train_epochs,
            "train_batch_size": training_args.per_device_train_batch_size,
            "n_embd": config.n_embd,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "vocab_size": config.vocab_size,
            "n_params": sum(p.numel() for p in model.parameters()),
        })

        trainer.train()
        trainer.save_model(out_dir)

        # Final eval on the best checkpoint: this is the number you compare across variants
        metrics = trainer.evaluate()
        eval_loss = metrics["eval_loss"]
        mlflow.log_metrics({
            "final_eval_loss": float(eval_loss),
            "final_eval_perplexity": math.exp(eval_loss),
        })

        logs = pd.DataFrame(trainer.state.log_history)
        log_path = os.path.join(log_dir, "pretrain_log.csv")
        logs.to_csv(log_path, index=False)
        mlflow.log_artifact(log_path, artifact_path="training")
        mlflow.log_artifacts(out_dir, artifact_path="model")

        if trainer.state.best_metric is not None:
            mlflow.log_metric("best_eval_metric", float(trainer.state.best_metric))

        print(f"[{ablation_name}] eval_loss={eval_loss:.4f}  run={run.info.run_id}")

    # Free GPU memory before the next variant
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return metrics


# def run_ablations(cfg, args):
#     """args.ablation: a key of ABLATIONS, or 'all'."""
#     names = list(ABLATIONS) if args.ablation == "all" else [args.ablation]
#     results = {}
#     for name in names:
#         results[name] = pretrain(cfg, args, name)
#     return results