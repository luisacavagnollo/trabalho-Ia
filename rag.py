"""
Módulo RAG (Retrieval-Augmented Generation) para o Hotel LuxStay.
Contém a base de conhecimento do hotel e funções de busca semântica.
Usa TF-IDF como método padrão (sem dependência externa).
Pode usar SentenceTransformers se disponível para melhor qualidade semântica.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Tenta importar sentence-transformers (opcional, para melhor qualidade)
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_DISPONIVEL = True
except ImportError:
    SENTENCE_TRANSFORMERS_DISPONIVEL = False

# ---------- Base de conhecimento do Hotel LuxStay ----------
BASE_CONHECIMENTO = [
    # Informações gerais
    {
        "categoria": "geral",
        "pergunta": "Qual o horário de check-in e check-out?",
        "resposta": "O check-in é a partir das 14h e o check-out até as 12h. Early check-in e late check-out estão sujeitos a disponibilidade e podem ter custo adicional de R$80,00 por hora."
    },
    {
        "categoria": "geral",
        "pergunta": "Qual o endereço do hotel?",
        "resposta": "O Hotel LuxStay está localizado na Av. Beira Mar, 1500 - Centro, Florianópolis/SC. Próximo ao Shopping Beiramar e a 15 minutos do Aeroporto Hercílio Luz."
    },
    {
        "categoria": "geral",
        "pergunta": "Quais formas de pagamento são aceitas?",
        "resposta": "Aceitamos cartões de crédito (Visa, Mastercard, Elo, Amex), débito, PIX e dinheiro. Para reservas online, aceitamos cartão de crédito e PIX."
    },
    {
        "categoria": "geral",
        "pergunta": "O hotel aceita animais de estimação?",
        "resposta": "Sim! Somos pet-friendly. Aceitamos animais de até 15kg com taxa adicional de R$50,00 por noite. É necessário informar no momento da reserva."
    },
    {
        "categoria": "geral",
        "pergunta": "Qual a política de cancelamento?",
        "resposta": "Cancelamentos com mais de 48h de antecedência são gratuitos. Entre 24h e 48h, cobramos 50% da primeira diária. Com menos de 24h, cobramos a primeira diária integral."
    },
    # Quartos e acomodações
    {
        "categoria": "quartos",
        "pergunta": "Quais tipos de quarto estão disponíveis?",
        "resposta": "Oferecemos: Quarto Standard (R$280/noite) - cama casal, TV, ar-condicionado, Wi-Fi; Quarto Superior (R$380/noite) - cama king, varanda, frigobar, TV 50'; Suíte Luxo (R$550/noite) - sala de estar, banheira, vista mar, minibar completo; Suíte Presidencial (R$950/noite) - 2 ambientes, jacuzzi, vista panorâmica, mordomo."
    },
    {
        "categoria": "quartos",
        "pergunta": "Os quartos possuem Wi-Fi?",
        "resposta": "Sim, todos os quartos e áreas comuns possuem Wi-Fi gratuito de alta velocidade (200 Mbps). A senha é fornecida no check-in."
    },
    {
        "categoria": "quartos",
        "pergunta": "Posso solicitar cama extra?",
        "resposta": "Sim, camas extras estão disponíveis por R$80,00 por noite, sujeito à disponibilidade e ao tipo de quarto. Berços para bebês são gratuitos."
    },
    {
        "categoria": "quartos",
        "pergunta": "O quarto tem cofre?",
        "resposta": "Sim, todos os quartos possuem cofre digital individual para guardar objetos de valor. O hotel não se responsabiliza por itens não guardados no cofre."
    },
    # Serviços e comodidades
    {
        "categoria": "servicos",
        "pergunta": "O hotel tem piscina?",
        "resposta": "Sim! Temos piscina ao ar livre com vista para o mar (aberta das 7h às 22h), piscina aquecida coberta (aberta das 6h às 23h) e piscina infantil."
    },
    {
        "categoria": "servicos",
        "pergunta": "Tem academia ou spa?",
        "resposta": "Sim! Nossa academia funciona 24h com equipamentos modernos. O Spa LuxStay oferece massagens, sauna, banho turco e tratamentos faciais (agendamento na recepção ou ramal 200)."
    },
    {
        "categoria": "servicos",
        "pergunta": "O café da manhã está incluso?",
        "resposta": "O café da manhã buffet está incluso em todas as tarifas. Servido das 6h30 às 10h30 (fins de semana até 11h). Temos opções veganas, sem glúten e sem lactose."
    },
    {
        "categoria": "servicos",
        "pergunta": "Tem estacionamento?",
        "resposta": "Sim, oferecemos estacionamento coberto com manobrista por R$40,00/dia para hóspedes. Vagas para veículos elétricos com carregador disponíveis."
    },
    {
        "categoria": "servicos",
        "pergunta": "Quais restaurantes o hotel possui?",
        "resposta": "Temos o Restaurante Mar Azul (almoço e jantar, cozinha contemporânea), o Lounge Bar (drinks e petiscos, 17h-01h) e o serviço de room service disponível 24h."
    },
    {
        "categoria": "servicos",
        "pergunta": "O hotel oferece transfer do aeroporto?",
        "resposta": "Sim! Oferecemos transfer aeroporto-hotel por R$90,00 (sedan) ou R$150,00 (van para grupos). Agende com 24h de antecedência pelo WhatsApp ou e-mail."
    },
    {
        "categoria": "servicos",
        "pergunta": "Tem lavanderia?",
        "resposta": "Sim, nosso serviço de lavanderia funciona de segunda a sábado. Roupas entregues até 9h ficam prontas no mesmo dia. Temos também lavanderia self-service no 2º andar."
    },
    {
        "categoria": "servicos",
        "pergunta": "O hotel tem sala de reuniões ou eventos?",
        "resposta": "Sim! Dispomos de 3 salas de reunião (10 a 50 pessoas) e 1 salão de eventos (até 200 pessoas). Equipamentos audiovisuais, coffee break e suporte técnico inclusos. Solicite orçamento na recepção."
    },
    # Turismo e localização
    {
        "categoria": "turismo",
        "pergunta": "Quais praias ficam próximas ao hotel?",
        "resposta": "As praias mais próximas são: Praia Central (5 min a pé), Praia do Müller (10 min de carro), Praia de Jurerê (25 min de carro) e Praia da Joaquina (30 min de carro)."
    },
    {
        "categoria": "turismo",
        "pergunta": "O hotel oferece passeios turísticos?",
        "resposta": "Sim! Na recepção organizamos: city tour por Florianópolis, passeio de barco pela Baía Norte, trilha ecológica na Lagoa do Peri e visita ao centro histórico. Consulte valores e horários."
    },
    {
        "categoria": "turismo",
        "pergunta": "Onde posso alugar um carro?",
        "resposta": "Temos parceria com a locadora MoveCar, com balcão no lobby do hotel. Também há locadoras no aeroporto. A recepção pode ajudar a fazer a reserva."
    },
    # Políticas e regras
    {
        "categoria": "politicas",
        "pergunta": "Qual a idade mínima para hospedagem?",
        "resposta": "Menores de 18 anos devem estar acompanhados de um responsável legal. A partir de 16 anos, é necessário apresentar autorização registrada em cartório."
    },
    {
        "categoria": "politicas",
        "pergunta": "É permitido fumar no hotel?",
        "resposta": "O Hotel LuxStay é 100% livre de fumo em áreas internas. Temos área externa designada para fumantes no terraço do 1º andar."
    },
    {
        "categoria": "politicas",
        "pergunta": "Existe taxa para visitantes?",
        "resposta": "Visitantes podem acessar o lobby e restaurantes sem custo. Para usar piscina e academia, há taxa de day-use de R$120,00 por pessoa."
    },
    # Contato e emergência
    {
        "categoria": "contato",
        "pergunta": "Como entrar em contato com o hotel?",
        "resposta": "Telefone: (48) 3333-1500 | WhatsApp: (48) 99999-1500 | E-mail: reservas@luxstay.com.br | Instagram: @hotelluxstay. Recepção 24h para hóspedes."
    },
    {
        "categoria": "contato",
        "pergunta": "O hotel tem serviço de concierge?",
        "resposta": "Sim! Nosso concierge está disponível das 7h às 23h para reservas em restaurantes, ingressos para eventos, dicas de passeios e qualquer necessidade especial."
    },
]

# ---------- Variáveis globais ----------
_modelo = None
_embeddings = None
_documentos = None
_carregado = False
_usar_sentence_transformers = False
_vectorizer = None


def _preparar_documentos():
    """Prepara os documentos da base de conhecimento para busca."""
    docs = []
    for item in BASE_CONHECIMENTO:
        texto = f"{item['pergunta']} {item['resposta']}"
        docs.append({
            "texto_busca": texto,
            "resposta": item["resposta"],
            "categoria": item["categoria"],
            "pergunta": item["pergunta"],
        })
    return docs


def carregar_base(status_callback=None, forcar_tfidf=False):
    """
    Carrega o modelo de embeddings e indexa a base de conhecimento.
    Tenta usar SentenceTransformers; se falhar, usa TF-IDF como fallback.
    """
    global _modelo, _embeddings, _documentos, _carregado, _usar_sentence_transformers, _vectorizer

    if status_callback:
        status_callback("Preparando documentos da base de conhecimento...")

    _documentos = _preparar_documentos()
    textos = [doc["texto_busca"] for doc in _documentos]

    # Tenta usar SentenceTransformers para melhor qualidade semântica
    if SENTENCE_TRANSFORMERS_DISPONIVEL and not forcar_tfidf:
        try:
            if status_callback:
                status_callback("Carregando modelo de embeddings (SentenceTransformers)...")

            _modelo = SentenceTransformer("all-MiniLM-L6-v2")
            _embeddings = _modelo.encode(textos, normalize_embeddings=True)
            _usar_sentence_transformers = True

            if status_callback:
                status_callback(f"Base carregada com SentenceTransformers! ({len(_documentos)} documentos)")

            _carregado = True
            return
        except Exception as e:
            if status_callback:
                status_callback(f"SentenceTransformers indisponível ({e}). Usando TF-IDF...")

    # Fallback: TF-IDF (funciona offline, sem download de modelos)
    if status_callback:
        status_callback("Indexando com TF-IDF...")

    _vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True,
    )
    _embeddings = _vectorizer.fit_transform(textos)
    _usar_sentence_transformers = False

    _carregado = True

    if status_callback:
        status_callback(f"Base carregada com TF-IDF! ({len(_documentos)} documentos)")


def esta_carregado():
    """Verifica se a base de conhecimento está carregada."""
    return _carregado


def buscar_contexto(pergunta, top_k=3):
    """
    Busca os documentos mais relevantes para a pergunta do usuário.
    Retorna o contexto formatado para o LLM.
    """
    if not _carregado:
        return "Base de conhecimento não carregada."

    if _usar_sentence_transformers:
        # Busca semântica com SentenceTransformers
        emb_pergunta = _modelo.encode([pergunta], normalize_embeddings=True)
        similaridades = np.dot(_embeddings, emb_pergunta.T).flatten()
    else:
        # Busca com TF-IDF
        emb_pergunta = _vectorizer.transform([pergunta])
        similaridades = cosine_similarity(emb_pergunta, _embeddings).flatten()

    # Pegar os top_k mais relevantes
    indices = np.argsort(similaridades)[::-1][:top_k]

    contexto_partes = []
    for i in indices:
        doc = _documentos[i]
        score = similaridades[i]
        if score > 0.05:  # Threshold mínimo de relevância
            contexto_partes.append(
                f"[{doc['categoria'].upper()}] P: {doc['pergunta']}\n"
                f"R: {doc['resposta']}"
            )

    if not contexto_partes:
        return "Nenhuma informação relevante encontrada na base de conhecimento."

    return "\n\n".join(contexto_partes)
