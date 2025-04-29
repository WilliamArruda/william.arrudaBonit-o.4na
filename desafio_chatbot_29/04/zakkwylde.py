def responder_como_zakk(pergunta):
    resposta_base = {
        "quem é você": "Sou Zakk Wylde, guitarrista da Black Label Society, ex-Ozzy, e servo das seis cordas!",
        "qual sua guitarra preferida": "Minha Gibson Les Paul 'The Grail', irmão. Ela ruge como um leão!",
        "dicas de guitarra": "Pratique até seus dedos sangrarem. E depois, pratique mais.",
        "metal": "Se não for alto, sujo e pesado... então nem é música.",
        "me ensina": "Se precisar de mais dicas e quiser aprender como eu toco, mande mensagem para esse cara: https://wa.me/5511910745074?text= Fala meu amigo!! Zakkbot me recomendou voçê para aprender mais sobre como tocar guitarra como ele"
    }

    pergunta = pergunta.lower()
    for chave in resposta_base:
        if chave in pergunta:
            return resposta_base[chave]
    
    return "HAHA! Boa pergunta, parceiro. Mas não tenho uma resposta pronta pra isso. Me pergunta sobre guitarras, Ozzy ou metal!"

def verificar_desejo_de_sair(texto):
    comandos_para_sair = ["sair", "exit", "tchau", "falou", "até mais", "valeu"]
    texto = texto.lower()
    return any(comando in texto for comando in comandos_para_sair)

def iniciar_chat():
    print("🎸 ZakkBot: Fala, irmão do metal! O que quer saber?")
    while True:
        user_input = input("Você: ")
        if verificar_desejo_de_sair(user_input):
            print("ZakkBot: Valeu, guerreiro das seis cordas! Que Odin te abençoe. 🤘")
            break
        resposta = responder_como_zakk(user_input)
        print("ZakkBot:", resposta)

iniciar_chat()
