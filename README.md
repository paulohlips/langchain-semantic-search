# Ingestão e Busca Semântica com LangChain e Postgres

Este projeto implementa um sistema de ingestão e busca semântica usando LangChain, OpenAI/Gemini e PostgreSQL com extensão pgVector.

## Funcionalidades

- **Ingestão**: Lê um arquivo PDF e salva suas informações em um banco de dados PostgreSQL com pgVector
- **Busca**: Permite fazer perguntas via CLI e recebe respostas baseadas apenas no conteúdo do PDF

## Pré-requisitos

- Python 3.8+
- Docker e Docker Compose
- Chave da API OpenAI ou Google (Gemini)

## Configuração

### 1. Clone o repositório e configure o ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure as variáveis de ambiente

Copie o arquivo `.env.example` para `.env` e preencha suas chaves de API:

```bash
cp .env.example .env
```

Edite o arquivo `.env` com suas chaves:

```env
OPENAI_API_KEY=sua-chave-openai-aqui
GOOGLE_API_KEY=sua-chave-google-aqui
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
GOOGLE_EMBEDDING_MODEL=models/embedding-001
PDF_PATH=./document.pdf
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=pdf_ingestion_collection
```

## Execução

### 1. Subir o banco de dados PostgreSQL

```bash
docker compose up -d
```

### 2. Executar ingestão do PDF

```bash
python src/ingest.py
```

### 3. Rodar o chat interativo

```bash
python src/chat.py
```

## Exemplo de uso

```
Digite sua pergunta ou 'exit' para sair.

Você: Qual o faturamento da Empresa SuperTechIABrazil?
Bot: O faturamento foi de 10 milhões de reais.

Você: Quantos clientes temos em 2024?
Bot: Não tenho informações necessárias para responder sua pergunta.

Você: exit
Encerrando chat...
```

## Tecnologias utilizadas

- **Python**: Linguagem de programação
- **LangChain**: Framework para aplicações com LLM
- **PostgreSQL + pgVector**: Banco de dados vetorial
- **OpenAI**: Embeddings (text-embedding-3-small) e LLM (gpt-5-nano)
- **Google Gemini**: Alternativa para embeddings (models/embedding-001)
- **Docker**: Para execução do banco de dados

## Estrutura do projeto

```
├── docker-compose.yml     # Configuração do PostgreSQL com pgVector
├── requirements.txt       # Dependências Python
├── .env.example          # Template das variáveis de ambiente
├── src/
│   ├── ingest.py         # Script de ingestão do PDF
│   ├── search.py         # Funções de busca semântica
│   └── chat.py           # Interface CLI para chat
├── document.pdf          # PDF para ingestão
└── README.md             # Este arquivo
```