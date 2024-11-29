#reminders #CUDA_VISIBLE_DEVICES=0 python colpali_FT.py
#conda activate colpali_FT
#!pip install --use-pep517 colpali-engine

from huggingface_hub import login
import wandb
from pathlib import Path
import torch
from colpali_engine.collators.visual_retriever_collator import VisualRetrieverCollator
from colpali_engine.loss import ColbertPairwiseCELoss
from colpali_engine.models import ColPali, ColPaliProcessor
from colpali_engine.trainer.contrastive_trainer import ContrastiveTrainer
from colpali_engine.utils.torch_utils import get_torch_device, tear_down_torch
from datasets import load_dataset
from torch import nn
from transformers import BitsAndBytesConfig, TrainingArguments

device = get_torch_device("auto")
#login() # optionnal HF login
wandb.login()


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || all params: {all_param:,} || trainable%: {100 * trainable_params / all_param}"

    )


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


model_name = "vidore/colpali-v1.3" #already has adapters merged
model = ColPali.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )


for name, param in model.named_parameters():
    if "lora" in name:
        param.requires_grad = True
        


processor = ColPaliProcessor.from_pretrained(model_name)
collator = VisualRetrieverCollator(processor=processor)


# Load the dataset
dataset_name = "CATIE-AQ/colpali_dataset"
ds = load_dataset(dataset_name)
ds = ds.rename_column("page_image", "image")
ds["train"] = ds["train"].shuffle(seed=42)

checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(exist_ok=True, parents=True)

training_args = TrainingArguments(
    output_dir=str(checkpoints_dir),
    hub_model_id="CATIE-AQ/finetune_colpali_pierre-4bit",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=False,
    eval_strategy="steps",
    save_steps=100000,
    logging_steps=100000,
    eval_steps=10000,
    warmup_steps=100,
    learning_rate=5e-5,
    save_total_limit=1,
    report_to=["wandb"] ,
)


trainer = ContrastiveTrainer(
    model=model,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    args=training_args,
    data_collator=collator,
    loss_func=ColbertPairwiseCELoss(),
    is_vision_model=True,
)

trainer.args.remove_unused_columns = False
eval_results = trainer.evaluate() #before training


run = wandb.init(
    project="colpali_FT",
    name="ft_4bit",
    config={
        "model_name": model_name,
        "bitsandbytes_config": bnb_config.to_dict(),
        "dataset_name": dataset_name,
    },
)

train_results = trainer.train()
eval_results = trainer.evaluate() #after training
run.finish()

trainer.push_to_hub("CATIE-AQ/colpali_FT_4bit_Pierre")

# Unload the previous model and clean the GPU cache
del model
tear_down_torch()

