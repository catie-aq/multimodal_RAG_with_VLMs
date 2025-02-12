Voici les scripts qui ont été produits pour entrainer les modèles nécessaires pour un pipeline de RAG multimodale en français.
Les datasets en français utilisés on été rassemblés et mis à disposition sur hugging face : https://huggingface.co/collections/CATIE-AQ/french-vqa-datasets-678a607a4c08258a5212950b et https://huggingface.co/collections/CATIE-AQ/french-retriever-datasets-678a6206c04f6a166139102c

- idefics3 finetuné en FR avec son script  
- colpali finetuné en FR avec son script 
Les poids des modèles sont ouverts sur hugging face :
https://huggingface.co/CATIE-AQ/idefics_fr_FT et https://huggingface.co/CATIE-AQ/finetune_colpali_pierre-4bit

Les techniques utilisées pour que cela puisse se faire sur une seule carte A100 : 
- flash attention 2
- adapters avec un QLoRA 
- quantization du modèle sur 8 bits
