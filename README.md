DONE :
- idefics3 finetuné en FR avec son script (sur une A100 : QLoRA + FA2) sur le dataset doc VQA_catie. Poids sur HF.

TO DO (dans l'ordre) :
- faire un vrai dataset d'évaluation avec des triplets (page de pdf, question, réponse)
- avoir un mécanisme qui dit si les réponses : "le prix est 22e", "22 euros" et "cela coûte 22 euros" sont les mêmes ou pas. Métrique : accuracy sur des réponses courtes et binaires type juste/faux.
- finetuner ColPali en Fr (avec le dataset doc VQA_catie)
- bencher le pipe colpali+idefics3 VS meilleure approche entre texte+OCR ou texte+captionning
- créer un dataset scrapé géant en FR avec la méthodo colpali (chercher des pdfs par thème et générer les Q/A ave un modèle type sonnet)
- ré-entrainer. Bencher le nouveau pipe.
