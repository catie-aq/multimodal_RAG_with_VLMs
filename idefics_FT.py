# conda activate essais_idefics
# launch command : CUDA_VISIBLE_DEVICES=1 python idefics_FT.py
# pour choisir le GPU

import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics3ForConditionalGeneration
from PIL import Image
from io import BytesIO
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

DEVICE = "cuda"
model_id = "HuggingFaceM4/Idefics3-8B-Llama3"

processor = AutoProcessor.from_pretrained(model_id)

#enable 8-bits quantization and QLoRA adapters 
lora_config = LoraConfig(
    r=8,
    lora_alpha=8,
    lora_dropout=0.1,
    target_modules='.*(text_model|connector).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
    use_dora=False,
    init_lora_weights="gaussian"
)
    
bnb_config = BitsAndBytesConfig(load_in_8bit=True) # to enable QLoRA

model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config,
    _attn_implementation="flash_attention_2",
)
model.add_adapter(lora_config) 
model.enable_adapters()

# freeze vision encoder
for param in model.model.vision_model.parameters():
    param.requires_grad = False

#dataset 
catie_vqa= load_dataset('CATIE-AQ/VQA-cmarkea-doc-vqa-clean', trust_remote_code=True, split="train")

# create trainer
def collate_fn(examples):
  texts = []
  images = []
  for example in examples:
      im_bytes = example["image"]
      image = Image.open(BytesIO(im_bytes))
      question = example["question"]
      answer = example["answer"] 
      messages = [
          {
              "role": "user",
              "content": [
                  {"type": "text", "text": "Answer briefly."},
                  {"type": "image"},
                  {"type": "text", "text": question}
              ]
          },
          {
              "role": "assistant",
              "content": [
                  {"type": "text", "text": answer}
              ]
          }
      ]
      
      text = processor.apply_chat_template(messages, add_generation_prompt=False) + processor.tokenizer.eos_token
      texts.append(text.strip())
      images.append([image])

  image_token_id = processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index("<image>")]
  batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
  labels = batch["input_ids"].clone()
  labels[labels == processor.tokenizer.pad_token_id] = image_token_id
  batch["labels"] = labels

  return batch



training_args = TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=1,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    save_strategy="steps",
    save_steps=40000,
    save_total_limit=1,
    optim="paged_adamw_8bit",
    bf16=True,
    output_dir="./my-idefics3-FT",
    hub_model_id="CATIE-AQ/idefics3_fr_FT",
    remove_unused_columns=False,
    push_to_hub=True
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=catie_vqa,
)

trainer.train()
print('training complete')

trainer.push_to_hub("CATIE-AQ/idefics3_french_FT")
print("pushed to hub")

