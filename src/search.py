from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_postgres import PGVector
import os
from dotenv import load_dotenv

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

load_dotenv()
for k in ("OPENAI_API_KEY", "PGVECTOR_URL"):
    if not os.getenv(k):
        raise RuntimeError(f"Environment variable {k} is not set")


embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_MODEL","text-embedding-3-small"))

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"),
    use_jsonb=True,
)

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
)


def search_prompt(question=None):
    if not question:
        return None

    docs = vector_store.similarity_search(question, k=10)

    contexto = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(
        contexto=contexto,
        pergunta=question
    )

    response = llm.invoke(prompt)

    return response.content