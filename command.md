# Contexto do Desafio

---

Estamos criando uma **solução digital para uma grande empresa** do setor financeiro que lida com um **alto volume de emails diariamente**. Esses emails podem ser mensagens solicitando um status atual sobre uma requisição em andamento, compartilhando algum arquivo ou até mesmo mensagens improdutivas, como desejo de feliz natal ou perguntas não relevantes. 

Nosso **objetivo é automatizar a leitura e classificação desses emails** e sugerir classificações e respostas automáticas de acordo com o teor de cada email recebido, **liberando tempo da equipe** para que não seja mais necessário ter uma pessoa fazendo esse trabalho manualmente.

## Objetivo do Desafio Simplificado

Desenvolver uma aplicação web simples que utilize inteligência artificial para:

1. **Classificar** emails em categorias predefinidas.
2. **Sugerir respostas automáticas** baseadas na classificação realizada.

**Categorias de Classificação**

- **Produtivo:** Emails que requerem uma ação ou resposta específica (ex.: solicitações de suporte técnico, atualização sobre casos em aberto, dúvidas sobre o sistema).
- **Improdutivo:** Emails que não necessitam de uma ação imediata (ex.: mensagens de felicitações, agradecimentos).

## Requisitos do Projeto

**1. Interface Web (HTML)**

**Formulário de Upload:**

- Permitir o upload de arquivos de email em formatos .txt ou .pdf ou a inserção direta de texto de emails.
- Botão para enviar o documento/email para processamento.

**Exibição dos Resultados:**

- Mostrar a categoria atribuída ao email (Produtivo ou Improdutivo).
- Exibir a resposta automática sugerida pelo sistema.

*Pro-tip:* a interface é uma ótima oportunidade para você se destacar, trazendo elementos visuais bem pensados, funcionalidades úteis, recursos mais avançados e uma experiência que realmente encante o usuário final.

**2. Backend em Python**

**Leitura e Processamento:**

- Desenvolver um script em Python que leia o conteúdo dos emails enviados.
- Utilizar técnicas de processamento de linguagem natural (NLP) para pré-processar o texto (remoção de stop words, stemming/lemmatização, etc.).

**Classificação e Resposta:**

- Implementar um algoritmo de classificação que categorize o conteúdo em **Produtivo** ou **Improdutivo**.
- Utilizar uma API de AI (como Hugging Face Transformers, OpenAI GPT, ou outra de sua preferência) para:

**Classificação:**

- Determinar a categoria do email.

**Geração de Resposta:**

- Sugerir uma resposta automática adequada à categoria identificada.

**Integração com a Interface Web:**

- Conectar o backend com a interface HTML para receber entradas e exibir resultados.

**3. Hospedagem na Nuvem**

**Deploy da Aplicação:**

- Hospedar a aplicação web em uma plataforma de nuvem gratuita (como Heroku, Vercel, AWS Free Tier, Google Cloud Platform, etc.) e disponibilizar um link online e funcional para que o time da AutoU ou usuários externos possam acessar e testar o funcionamento da solução.
- Fornecer o link para a aplicação hospedada junto com a submissão do desafio.

---

## Entregáveis

**1. Código Fonte:**

Forneça o **link público** para o repositório (GitHub) contendo:

- Scripts em Python (.py, .ipynb)
- Arquivo(s) HTML ou outros arquivos da interface
- Arquivo requirements.txt (ou similar)
- Dados de exemplo (caso necessário)
- Arquivo README no repositório com instruções claras de como executar sua aplicação localmente e outras informações úteis (se aplicável)
- Qualquer outro material relevante para entender e rodar o projeto

> *Obs*: Organize o repositório com atenção à estrutura e à clareza de leitura.
> 

**2. Vídeo Demonstrativo:**

Grave um vídeo de **3 a 5 minutos** apresentando sua solução e envie o link (YouTube com acesso liberado).

**Conteúdo do Vídeo:**

- **Introdução (30 segundos):** Apresentação pessoal e breve descrição do desafio.
- **Demonstração (3 minutos):** Mostrar a interface web, realizar o upload de um email, e exibir a classificação e a resposta sugerida.
    - **Explicação Técnica (1 minuto):** Descrever brevemente como o algoritmo funciona, as tecnologias usadas, o processo de treinamento da AI e as principais decisões técnicas.
- **Conclusão (30 segundos):** Resumir o que foi feito e destacar pontos de aprendizado.

> *Obs*: Garanta que seu vídeo está publicado e com acesso liberado para que qualquer um com o link possa assistir.
> 

**3. Link da Solução Deployada na Nuvem**

Forneça um **link funcional** para que possamos acessar e testar sua aplicação diretamente em um **ambiente online** (por exemplo: Vercel, Render, Hugging Face Spaces, Replit, Heroku, GCP, Azure, AWS, etc).

A aplicação deve estar pronta para uso, sem necessidade de instalação local. Caso haja autenticação ou instruções específicas para uso, inclua essa orientação na tela inicial ou no próprio README enviado no Github.

A interface deve ser simples, intuitiva e com uma navegação amigável, mesmo para usuários não técnicos. Recomendamos que o foco esteja em:

- **Facilidade de uso** (sem dependência de instruções externas);
- **Clareza na experiência** (mostrar o propósito da aplicação de forma direta);
- **Organização visual** (sem excesso de elementos ou distrações desnecessárias).

Caso opte por não hospedar a aplicação, justifique brevemente o motivo no campo de entrega, e certifique-se de que o repositório contenha todos os arquivos necessários para rodar localmente, com instruções claras.

---

**Critérios de Avaliação**

**1. Funcionalidade e experiência do usuário:**

- A aplicação realiza a classificação correta dos emails nas categorias **Produtivo** e **Improdutivo**.
- A resposta sugerida é relevante e adequada para a categoria identificada.
- A experiência do usuário com a plataforma é fluída e intuitiva.

**2. Qualidade Técnica:**

- Código limpo, organizado e bem documentado.
- Uso eficaz das bibliotecas e APIs de AI.

**3. Uso de AI:**

- Integração correta e eficaz das APIs de NLP para classificação e geração de respostas.
- Demonstração de treinamento ou ajuste da AI para melhorar a qualidade das respostas.

**4. Hospedagem na Nuvem:**

- A aplicação está hospedada e acessível via uma URL fornecida.

- Funcionamento consistente e sem erros na aplicação hospedada.

**5. Interface Web (HTML):**

- Interface funcional e intuitiva para upload de arquivos e exibição de resultados.
- (Extra) Capricho visual ou recursos adicionais que melhorem a usabilidade.

6. **Autonomia e Resolução de Problemas:**

- Capacidade de resolver problemas técnicos de forma independente.
- Proatividade na busca de soluções e implementação eficiente.

7. **Demonstração e Comunicação:**

- Clareza e concisão na apresentação do vídeo.
- Capacidade de explicar o funcionamento do algoritmo, o processo de treinamento da AI e as escolhas técnicas de forma compreensível.

---

## Instruções de Entrega

**A entrega deve ser feita exclusivamente através do formulário:** https://airtable.com/appSpmPaaJ41PwbiU/pageVZHk7NuMhfR9J/form**.**

No formulário, você irá colar os seguintes links:

- Link público do repositório no GitHub com o código-fonte do projeto e outras informações úteis.
- Link público do vídeo de apresentação rápida do projeto (ex: YouTube).
- Link da aplicação publicada (ex: Render, Hugging Face Spaces, Replit, Heroku, GCP, AWS, Azure ou similar).

1. Certifique-se de que todos os **links estejam acessíveis publicamente** e que não exijam permissões especiais.
2. O **envio deve ser feito até a data limite informada no e-mail** de confirmação. Envios após o prazo não serão considerados.
3. O envio do case deve ser feito **exclusivamente pelo formulário** acima. **Não serão aceitos arquivos ou links enviados por e-mail, WhatsApp ou qualquer outro canal.** Toda a análise será feita apenas com base nas informações preenchidas no formulário.

---

## Considerações finais

**Minha principal dica para você é:**

Enxergar este case como uma grande oportunidade de aplicar seus conhecimentos na criação de algo prático e se desenvolver profissionalmente. Nós forneceremos feedbacks sobre os cases, o que será uma excelente forma de você evoluir na sua carreira.

Foque em realmente entender o desafio, o contexto do problema e em criar uma solução digital que resolva e melhore a vida do usuário. Não existe resposta certa ou errada, nem gabarito. 

**Capriche bastante na sua solução, estamos ansiosos para ver o que você irá criar, boa sorte!** 🚀