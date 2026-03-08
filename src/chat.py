from search import search_prompt


def main():
    print("Digite sua pergunta ou 'exit' para sair.\n")

    while True:
        question = input("Você: ")

        if question.lower() in ["exit", "quit", "sair"]:
            print("Encerrando chat...")
            break

        try:
            response = search_prompt(question)

            if not response:
                print("Bot: Não foi possível gerar resposta.\n")
                continue

            print(f"Bot: {response}\n")

        except Exception as e:
            print(f"Erro ao processar pergunta: {e}\n")


if __name__ == "__main__":
    main()