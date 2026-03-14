---
title: Email Classification API
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Email Classification API

Classifica e-mails corporativos como **Produtivo** ou **Improdutivo** usando IA (BART zero-shot) e gera respostas automáticas com Qwen 72B.

## Funcionalidades
- Classificação automática via NLP (facebook/bart-large-mnli)
- Geração de respostas com IA (Qwen/Qwen2.5-72B-Instruct)
- Upload de arquivos .txt e .pdf
- Interface web single-page

## Configuração
Defina o secret `HUGGINGFACE_API_TOKEN` nas configurações do Space para habilitar a geração de respostas com IA.
