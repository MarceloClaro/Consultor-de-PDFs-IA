import os
import pdfplumber
import json
import re
import pandas as pd
import streamlit as st
from typing import Tuple
import time
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq

# Configurações da página do Streamlit
st.set_page_config(
    page_title="Consultor de PDFs + IA",
    page_icon="logo.png",
    layout="wide",
)

# Definição de constantes
FILEPATH = "agents.json"
CHAT_HISTORY_FILE = 'chat_history.json'
API_USAGE_FILE = 'api_usage.json'

MODEL_MAX_TOKENS = {
    'mixtral-8x7b-32768': 32768,
    'llama3-70b-8192': 8192,
    'llama3-8b-8192': 8192,
    'gemma-7b-it': 8192,
}

# Definição das chaves de API
API_KEYS = {
    "fetch": ["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", "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"],
    "refine": ["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", "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"],
    "evaluate": ["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", "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"]
}

# Variáveis para manter o estado das chaves de API
CURRENT_API_KEY_INDEX = {
    "fetch": 0,
    "refine": 0,
    "evaluate": 0
}

# Função para obter a próxima chave de API disponível
def get_next_api_key(action: str) -> str:
    global CURRENT_API_KEY_INDEX
    keys = API_KEYS.get(action, [])
    if keys:
        key_index = CURRENT_API_KEY_INDEX[action]
        api_key = keys[key_index]
        CURRENT_API_KEY_INDEX[action] = (key_index + 1) % len(keys)
        return api_key
    else:
        raise ValueError(f"No API keys available for action: {action}")

# Função para manipular limites de taxa
def handle_rate_limit(error_message: str, action: str):
    wait_time = 80  # Tempo padrão de espera
    match = re.search(r'Aguardando (\d+\.?\d*) segundos', error_message)
    if match:
        wait_time = float(match.group(1))
    st.warning(f"Limite de taxa atingido. Aguardando {wait_time} segundos...")
    time.sleep(wait_time)
    # Alterna a chave de API para a próxima disponível
    new_key = get_next_api_key(action)
    st.info(f"Usando nova chave de API para {action}: {new_key}")

def load_agent_options() -> list:
    agent_options = ['Escolher um especialista...']
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r') as file:
            try:
                agents = json.load(file)
                agent_options.extend([agent["agente"] for agent in agents if "agente" in agent])
            except json.JSONDecodeError:
                st.error("Erro ao ler o arquivo de Agentes. Por favor, verifique o formato.")
    return agent_options

def extrair_texto_pdf(file):
    texto_paginas = []
    with pdfplumber.open(file) as pdf:
        for num_pagina in range(len(pdf.pages)):
            pagina = pdf.pages[num_pagina]
            texto_pagina = pagina.extract_text()
            if texto_pagina:
                texto_paginas.append({'page': num_pagina + 1, 'text': texto_pagina})
    return texto_paginas

def text_to_dataframe(texto_paginas):
    dados = {'Page': [], 'Text': []}
    for entrada in texto_paginas:
        dados['Page'].append(entrada['page'])
        dados['Text'].append(entrada['text'])
    return pd.DataFrame(dados)

def identificar_secoes(texto, secao_inicial):
    secoes = {}
    secao_atual = secao_inicial
    secoes[secao_atual] = ""

    paragrafos = texto.split('\n')
    for paragrafo in paragrafos:
        match = re.match(r'Parte \d+\.', paragrafo) or re.match(r'Capítulo \d+: .*', paragrafo) or re.match(r'\d+\.\d+ .*', paragrafo)
        if match:
            secao_atual = match.group()
            secoes[secao_atual] = ""
        else:
            secoes[secao_atual] += paragrafo + "\n"

    return secoes

def salvar_como_json(dados, caminho_saida):
    with open(caminho_saida, 'w', encoding='utf-8') as file:
        json.dump(dados, file, ensure_ascii=False, indent=4)

def processar_e_salvar(texto_paginas, secao_inicial, caminho_pasta_base, nome_arquivo):
    secoes = identificar_secoes(" ".join([entrada['text'] for entrada in texto_paginas]), secao_inicial)
    caminho_saida = os.path.join(caminho_pasta_base, f"{nome_arquivo}.json")
    salvar_como_json(secoes, caminho_saida)

def preencher_dados_faltantes(titulo):
    return {
        'titulo': titulo,
        'autor': 'Autor Desconhecido',
        'ano': 'Ano Desconhecido',
        'paginas': 'Páginas Desconhecidas'
    }

def upload_and_extract_references(uploaded_file):
    references = {}
    try:
        if uploaded_file.name.endswith('.json'):
            references = json.load(uploaded_file)
            with open("references.json", 'w') as file:
                json.dump(references, file, indent=4)
            return "references.json"
        elif uploaded_file.name.endswith('.pdf'):
            texto_paginas = extrair_texto_pdf(uploaded_file)
            if not texto_paginas:
                st.error("Nenhum texto extraído do PDF.")
                return pd.DataFrame()
            df = text_to_dataframe(texto_paginas)
            if not df.empty:
                df.to_csv("references.csv", index=False)
                return df
            else:
                st.error("Nenhum texto extraído do PDF.")
                return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro ao carregar e extrair referências: {e}")
        return pd.DataFrame()

def get_max_tokens(model_name: str) -> int:
    return MODEL_MAX_TOKENS.get(model_name, 4096)

def log_api_usage(action: str, interaction_number: int, tokens_used: int, time_taken: float, user_input: str, user_prompt: str, api_response: str, agent_used: str, agent_description: str):
    entry = {
        'action': action,
        'interaction_number': interaction_number,
        'tokens_used': tokens_used,
        'time_taken': time_taken,
        'user_input': user_input,
        'user_prompt': user_prompt,
        'api_response': api_response,
        'agent_used': agent_used,
        'agent_description': agent_description
    }
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r+') as file:
            api_usage = json.load(file)
            api_usage.append(entry)
            file.seek(0)
            json.dump(api_usage, file, indent=4)
    else:
        with open(API_USAGE_FILE, 'w') as file:
            json.dump([entry], file, indent=4)

def save_chat_history(user_input, user_prompt, expert_response, chat_history_file=CHAT_HISTORY_FILE):
    chat_entry = {
        'user_input': user_input,
        'user_prompt': user_prompt,
        'expert_response': expert_response
    }
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r+') as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
            chat_history.append(chat_entry)
            file.seek(0)
            json.dump(chat_history, file, indent=4)
    else:
        with open(chat_history_file, 'w') as file:
            json.dump([chat_entry], file, indent=4)

def load_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        with open(chat_history_file, 'r') as file:
            try:
                chat_history = json.load(file)
            except json.JSONDecodeError:
                chat_history = []
        return chat_history
    return []

def clear_chat_history(chat_history_file=CHAT_HISTORY_FILE):
    if os.path.exists(chat_history_file):
        os.remove(chat_history_file)

def load_api_usage():
    if os.path.exists(API_USAGE_FILE):
        with open(API_USAGE_FILE, 'r') as file:
            try:
                api_usage = json.load(file)
            except json.JSONDecodeError:
                api_usage = []
        return api_usage
    return []

def plot_api_usage(api_usage):
    df = pd.DataFrame(api_usage)

    if 'action' not in df.columns:
        st.error("A coluna 'action' não foi encontrada no dataframe de uso da API.")
        return

    if 'agent_description' in df.columns:
        df['agent_description'] = df['agent_description'].apply(lambda x: json.dumps(x) if isinstance(x, dict) else str(x))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    sns.histplot(df[df['action'] == 'fetch']['tokens_used'], bins=20, color='blue', label='Fetch', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'refine']['tokens_used'], bins=20, color='green', label='Refine', ax=ax1, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['tokens_used'], bins=20, color='red', label='Evaluate', ax=ax1, kde=True)
    ax1.set_title('Uso de Tokens por Chamada de API')
    ax1.set_xlabel('Tokens')
    ax1.set_ylabel('Frequência')
    ax1.legend()

    sns.histplot(df[df['action'] == 'fetch']['time_taken'], bins=20, color='blue', label='Fetch', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'refine']['time_taken'], bins=20, color='green', label='Refine', ax=ax2, kde=True)
    sns.histplot(df[df['action'] == 'evaluate']['time_taken'], bins=20, color='red', label='Evaluate', ax=ax2, kde=True)
    ax2.set_title('Tempo por Chamada de API')
    ax2.set_xlabel('Tempo (s)')
    ax2.set_ylabel('Frequência')
    ax2.legend()

    st.sidebar.pyplot(fig)

    st.sidebar.markdown("### Uso da API - DataFrame")
    st.sidebar.dataframe(df)

def reset_api_usage():
    if os.path.exists(API_USAGE_FILE):
        os.remove(API_USAGE_FILE)
    st.success("Os dados de uso da API foram resetados.")

def fetch_assistant_response(user_input: str, user_prompt: str, model_name: str, temperature: float, agent_selection: str, chat_history: list, interaction_number: int, references_df: pd.DataFrame = None) -> Tuple[str, str]:
    phase_two_response = ""
    expert_title = ""
    expert_description = ""
    try:
        client = Groq(api_key=get_next_api_key('fetch'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            backoff_time = 1
            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é um assistente útil."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=get_max_tokens(model_name),
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content if completion.choices else ""
                    log_api_usage('fetch', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description)
                    return api_response
                except Exception as e:
                    if "503" in str(e):
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")
                        return ""
                    handle_rate_limit(str(e), 'fetch')
                    backoff_time = min(backoff_time * 2, 84)
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")
                    time.sleep(backoff_time)

        if agent_selection == "Escolher um especialista...":
            phase_one_prompt = (
                f"\n\n<solicitacoes>\n"
                f"0 - 用葡萄牙语书写。10 个段落，每个段落 5 个句子，每个句子一个逗号。\n"
                f"1 - TÍTULO - PDF 的首句话通常是标题，标题中的每个单词通常以大写字母开头。很可能是 PDF 的第一句话。如果有必要，请通读整篇文章，看看是否能找到带有文章名称的参考文献，或者在网上查找参考文献以提取标题。\n"
                f"2 - OBJETIVOS - 识别目标：仔细分析工作的内容，以提取主要目标。将这些目标总结为一个清晰简洁的段落，捕捉研究的本质和意图。\n"
                f"3 - GAP - 识别差距：分析工作的内容，以识别其涉及的科学差距，即使未明确提及。简明扼要地总结段落，重点突出研究试图解决或阐明的核心问题。\n"
                f"4 - METODOLOGIA - 提取详细的工作方法：识别和描述方法：仔细分析工作，以识别所使用的方法。详细描述方法的各个方面，包括研究设计、使用的技术和工具、数据收集和分析程序、方法步骤以及采用的任何特定或创新方法。将描述限制在 250 字以内，以保持简洁而不牺牲重要细节。\n"
                f"5 - DATASET - 识别工作中使用的数据集。简要描述，限制为 40 字。同一行列出数据集名称并用逗号分隔。如果数据集是作者创建的，则写“OWN DATASET”。\n"
                f"6 - RESULTADOS - 用一个段落描述研究结果，强调定量数据，并明确提供数值数据。这个段落还应强调与先前最好的研究相比，新工作的结果。不要使用最高级。保持中立和科学的语气。\n"
                f"7 - LIMITAÇÕES - 转述研究的局限性。\n"
                f"8 - CONCLUSÃO - 总结作者对工作的结论。\n"
                f"9 - FUTURO - 提取对未来研究的建议：根据文章的结论指出对未来研究的建议。\n"
                f"10 - AVALIAÇÃO - 对工作进行批判性评价。不要泛泛而谈，要深入分析。\n"
                f"</solicitacoes>\n"
                f"\n\n<controle>\n"
                f"控制级别：\n"
                f"1. 语气：正式科学。\n"
                f"2. 主题焦点：您应始终高度关注科学文章的文本。\n"
                f"3. 语言：始终以巴西葡萄牙语回答，如巴西人通常撰写科学文章一样，遵循该国的科学写作标准，除非另有说明。\n"
                f"4. 情感控制：中立和科学。避免使用最高级，如：创新、革命性等。\n"
                f"5. 原创性水平：10，其中 1 是低原创性，10 是高原创性。绝不要复制原文中的句子。\n"
                f"6. 抽象水平：1，其中 1 是非常具体和真实，10 是非常抽象和不真实。\n"
                f"7. 动词时态：使用过去时。\n"
                f"</controle>\n"
                f"\n\n<restricoes>\n"
                f"不应从英语翻译成葡萄牙语的内容：\n"
                f"1. 在葡萄牙语文章中广泛接受和使用的英语技术术语。\n"
                f"2. 机器学习算法的名称。\n"
                f"3. 工作中使用的指标。\n"
                f"4. 数据集名称。\n"
                f"5. 不要将 YAML 的返回值包含在 ```yaml 中。\n"
                f"6. 不要在返回文本中包含 ``` 或 ```。\n"
                f"</restricoes>\n"
                f"\n\n<template>\n"
                f"文章：\n"
                f"  - 标题：\"文章标题\"\n"
                f"  - 文件：\"文件名.pdf\"\n"
                f"  - 目标：\"总体目标和具体目标\"\n"
                f"  - 差距：\"科学差距\"\n"
                f"  - 方法：\"方法\"\n"
                f"  - 数据集：\"使用的数据集\"\n"
                f"  - 结果：\"文章结果\"\n"
                f"  - 限制：\"科学文章的限制\"\n"
                f"  - 结论：\"结论\"\n"
                f"  - 评价：\"文章分析\"\n"
                f"</template>\n"
            )


            phase_one_response = get_completion(phase_one_prompt)
            first_period_index = phase_one_response.find(".")
            if first_period_index != -1:
                expert_title = phase_one_response[:first_period_index].strip()
                expert_description = phase_one_response[first_period_index + 1:].strip()
                save_expert(expert_title, expert_description)
            else:
                st.error("Erro ao extrair título e descrição do especialista.")
        else:
            if os.path.exists(FILEPATH):
                with open(FILEPATH, 'r') as file:
                    agents = json.load(file)
                    agent_found = next((agent for agent in agents if agent["agente"] == agent_selection), None)
                    if agent_found:
                        expert_title = agent_found["agente"]
                        expert_description = agent_found["descricao"]
                    else:
                        raise ValueError("Especialista selecionado não encontrado no arquivo.")
            else:
                raise FileNotFoundError(f"Arquivo {FILEPATH} não encontrado.")

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        references_context = ""
        if references_df is not None:
            for index, row in references_df.iterrows():
                titulo = row.get('titulo', 'Título Desconhecido')
                autor = row.get('autor', 'Autor Desconhecido')
                ano = row.get('ano', 'Ano Desconhecido')
                paginas = row.get('Page', 'Página Desconhecida')
                references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPáginas: {paginas}\n\n"

        phase_two_prompt = (
                f"{expert_title}, 请完整、详细并且必须用葡萄牙语回答以下请求：{user_input} 和 {user_prompt}。"
                f"\n\n聊天记录：{history_context}"
                f"\n\n参考资料：\n{references_context}"
                f"\n\n详细回答说明：\n"
                f"请完整、详细并且必须用葡萄牙语回答以下请求。请确保涉及所有相关方面并提供清晰准确的信息。使用示例、数据和额外解释来丰富回答。结构化回答，使其逻辑清晰、易于理解。\n"
                f"请求：\n"
                f"{user_input}\n"
                f"{user_prompt}\n"
                f"\n回答标准：\n"
                f"1. 引言：概述主题，并说明请求的背景。\n"
                f"2. 详细说明：详细解释请求的每个相关方面。使用小标题来组织信息，方便阅读。\n"
                f"3. 示例和数据：包括实际示例、案例研究、统计数据或相关数据来说明所提到的要点。\n"
                f"4. 批判性分析：对提供的数据和信息进行批判性分析，突出其意义、好处和可能的挑战。\n"
                f"5. 结论：总结回答的主要要点，并提出明确、客观的结论。\n"
                f"6. 参考资料：如果适用，请引用在回答中使用的来源和参考文献。\n"
                f"\n结构示例：\n"
                f"1. 引言\n"
                f"- 主题背景\n"
                f"- 主题重要性\n"
                f"2. 相关方面\n"
                f"- 小标题 1\n"
                f"  - 小标题 1 的详细说明\n"
                f"  - 示例和数据\n"
                f"- 小标题 2\n"
                f"  - 小标题 2 的详细说明\n"
                f"  - 示例和数据\n"
                f"3. 批判性分析\n"
                f"- 提供数据的讨论\n"
                f"- 意义和挑战\n"
                f"4. 结论\n"
                f"- 主要要点总结\n"
                f"- 明确结论\n"
                f"5. 参考资料\n"
                f"- 来源和参考文献列表\n"
                f"\n请使用此格式确保回答全面、信息丰富且结构良好。在发送前，请务必审查和编辑回答以确保清晰和准确。\n"
                f"\n---\n"
                f"\ngen_id: [自动生成]\n"
                f"seed: [自动生成]\n"
                f"seed: [gerado automaticamente]\n"
        )
        phase_two_response = get_completion(phase_two_prompt)

    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")
        return "", ""

    return expert_title, phase_two_response

def refine_response(expert_title: str, phase_two_response: str, user_input: str, user_prompt: str, model_name: str, temperature: float, references_context: str, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_next_api_key('refine'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            backoff_time = 1
            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é一个 assistente útil."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=get_max_tokens(model_name),
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content if completion.choices else ""
                    log_api_usage('refine', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, "")
                    return api_response
                except Exception as e:
                    if "503" in str(e):
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")
                        return ""
                    handle_rate_limit(str(e), 'refine')
                    backoff_time = min(backoff_time * 2, 64)
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")
                    time.sleep(backoff_time)

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        refine_prompt = (
            f"{expert_title}, 请完善以下回答：{phase_two_response}。原始请求：{user_input} 和 {user_prompt}。"
            f"\n\n聊天记录：{history_context}"
            f"\n\n参考资料：\n{references_context}"
            f"\n\n回答优化说明：\n"
            f"请优化提供的回答，确保其更加完整和详细。请确保涉及所有相关方面，并提供清晰准确的信息。使用更多的示例、数据和补充说明进一步丰富回答。将回答结构化，使其逻辑清晰，易于理解。\n"
            f"\n优化标准：\n"
            f"1. 引言：确保引言概述主题，并说明请求的背景。\n"
            f"2. 详细说明：核查并扩展请求的每个相关方面，使用小标题来组织信息并便于阅读。\n"
            f"3. 示例和数据：增加更多实际示例、案例研究、统计数据或相关数据，以说明所提到的要点。\n"
            f"4. 批判性分析：深入分析提供的数据和信息，突出其意义、好处和可能的挑战。\n"
            f"5. 结论：审查并强化回答的主要要点，提出明确、客观的结论。\n"
            f"6. 参考资料：增加任何可能用来撰写回答的额外来源和参考文献。\n"
            f"\n请使用此格式确保优化后的回答更加全面、信息丰富且结构良好。在发送前，请务必审查和编辑回答以确保清晰和准确。\n"
            f"\n---\n"
            f"\ngen_id: [自动生成]\n"
            f"seed: [自动生成]\n"
            f"seed: [gerado automaticamente]\n"
        )

        if not references_context:
            refine_prompt += (
                f"\n\nDevido à ausência de referências fornecidas, certifique-se de fornecer uma resposta detalhada, precisa e obrigatoriamente em português:, mesmo sem o uso de fontes externas."
                f"{expert_title}, 请完善以下回答：{phase_two_response}。原始请求：{user_input} 和 {user_prompt}。"
                f"\n\n聊天记录：{history_context}"
                f"\n\n参考资料：\n{references_context}"
                f"\n\n回答优化说明：\n"
                f"请优化提供的回答，确保其更加完整和详细。请确保涉及所有相关方面，并提供清晰准确的信息。使用更多的示例、数据和补充说明进一步丰富回答。将回答结构化，使其逻辑清晰，易于理解。\n"
                f"\n优化标准：\n"
                f"1. 引言：确保引言概述主题，并说明请求的背景。\n"
                f"2. 详细说明：核查并扩展请求的每个相关方面，使用小标题来组织信息并便于阅读。\n"
                f"3. 示例和数据：增加更多实际示例、案例研究、统计数据或相关数据，以说明所提到的要点。\n"
                f"4. 批判性分析：深入分析提供的数据和信息，突出其意义、好处和可能的挑战。\n"
                f"5. 结论：审查并强化回答的主要要点，提出明确、客观的结论。\n"
                f"6. 参考资料：增加任何可能用来撰写回答的额外来源和参考文献。\n"
                f"\n请使用此格式确保优化后的回答更加全面、信息丰富且结构良好。在发送前，请务必审查和编辑回答以确保清晰和准确。\n"
                f"\n---\n"
                f"\ngen_id: [自动生成]\n"
                f"seed: [自动生成]\n"
            )

        refined_response = get_completion(refine_prompt)
        return refined_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante o refinamento: {e}")
        return ""

def evaluate_response_with_rag(user_input: str, user_prompt: str, expert_title: str, expert_description: str, assistant_response: str, model_name: str, temperature: float, chat_history: list, interaction_number: int) -> str:
    try:
        client = Groq(api_key=get_next_api_key('evaluate'))

        def get_completion(prompt: str) -> str:
            start_time = time.time()
            backoff_time = 1
            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Você é一个 assistente útil."},
                            {"role": "user", "content": prompt},
                        ],
                        model=model_name,
                        temperature=temperature,
                        max_tokens=get_max_tokens(model_name),
                        top_p=1,
                        stop=None,
                        stream=False
                    )
                    end_time = time.time()
                    tokens_used = completion.usage.total_tokens
                    time_taken = end_time - start_time
                    api_response = completion.choices[0].message.content if completion.choices else ""
                    log_api_usage('evaluate', interaction_number, tokens_used, time_taken, user_input, user_prompt, api_response, expert_title, expert_description)
                    return api_response
                except Exception as e:
                    if "503" in str(e):
                        st.error(f"Ocorreu um erro: Error code: 503 - {e}")
                        return ""
                    handle_rate_limit(str(e), 'evaluate')
                    backoff_time = min(backoff_time * 2, 64)
                    st.warning(f"Limite de taxa atingido. Aguardando {backoff_time} segundos...")
                    time.sleep(backoff_time)

        history_context = ""
        for entry in chat_history:
            history_context += f"\nUsuário: {entry['user_input']}\nEspecialista: {entry['expert_response']}\n"

        rag_prompt = (
            f"\n\nCertifique-se de fornecer uma resposta detalhada, precisa e obrigatoriamente em português:, mesmo sem o uso de fontes externas."
            f"{expert_title}, 请评估以下回答：{assistant_response}。原始请求：{user_input} 和 {user_prompt}。"
            f"\n\n聊天记录：{history_context}"
            f"\n\n详细描述提供的回答中的可能改进点，并且必须用葡萄牙语：\n"
            f"\n回答评估和改进说明：\n"
            f"请使用以下分析和方法评估提供的回答：\n"
            f"1. SWOT 分析：识别回答中的优势、劣势、机会和威胁。\n"
            f"2. Q-统计分析：评估回答中所提供信息的统计质量。\n"
            f"3. Q-指数分析：检查提供数据的相关性和指数适用性。\n"
            f"4. PESTER 分析：考虑回答中涉及的政治、经济、社会、技术、生态和监管因素。\n"
            f"5. 连贯性：评估回答的连贯性，检查文本部分之间的流畅性和连接性。\n"
            f"6. 逻辑性：检查回答的逻辑一致性，确保信息结构合理，整体连贯。\n"
            f"7. 流畅性：分析文本的流畅性，确保阅读过程轻松愉快。\n"
            f"8. 差距分析：识别回答中可以进一步发展或澄清的差距或领域。\n"
            f"\n根据这些分析提供详细的改进建议，确保最终回答全面、准确且结构良好。\n"
            f"\n---\n"
            f"\ngen_id: [自动生成]\n"
            f"seed: [自动生成]\n"
        )

        rag_response = get_completion(rag_prompt)
        return rag_response

    except Exception as e:
        st.error(f"Ocorreu um erro durante a avaliação com RAG: {e}")
        return ""

def save_expert(expert_title: str, expert_description: str):
    new_expert = {
        "agente": expert_title,
        "descricao": expert_description
    }
    if os.path.exists(FILEPATH):
        with open(FILEPATH, 'r+') as file:
            try:
                agents = json.load(file)
            except json.JSONDecodeError:
                agents = []
            agents.append(new_expert)
            file.seek(0)
            json.dump(agents, file, indent=4)
    else:
        with open(FILEPATH, 'w') as file:
            json.dump([new_expert], file, indent=4)

# Interface Principal com Streamlit

if 'resposta_assistente' not in st.session_state:
    st.session_state.resposta_assistente = ""
if 'descricao_especialista_ideal' not in st.session_state:
    st.session_state.descricao_especialista_ideal = ""
if 'resposta_refinada' not in st.session_state:
    st.session_state.resposta_refinada = ""
if 'resposta_original' not in st.session_state:
    st.session_state.resposta_original = ""
if 'rag_resposta' not in st.session_state:
    st.session_state.rag_resposta = ""
if 'references_df' not in st.session_state:
    st.session_state.references_df = pd.DataFrame()

agent_options = load_agent_options()

st.image('updating (2).gif', width=100, caption='Laboratório de Educação e Inteligência Artificial - Geomaker. "A melhor forma de prever o futuro é inventá-lo." - Alan Kay', use_column_width='always', output_format='auto')
st.markdown("<h1 style='text-align: center;'>Consultor de PDFs</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Utilize nossa plataforma para estudos em diversos assuntos com PDF.</h2>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

memory_selection = st.selectbox("Selecione a quantidade de interações para lembrar:", options=[5, 10, 15, 25, 50, 100, 150, 300, 450])

st.write("Digite sua solicitação para que ela seja respondida pelo especialista ideal.")
col1, col2 = st.columns(2)

with col1:
    user_input = st.text_area("Por favor, insira sua solicitação:", height=200, key="entrada_usuario")
    user_prompt = st.text_area("Escreva um prompt ou coloque o texto para consulta para o especialista (opcional):", height=200, key="prompt_usuario")
    agent_selection = st.selectbox("Escolha um Especialista", options=agent_options, index=0, key="selecao_agente")
    model_name = st.selectbox("Escolha um Modelo", list(MODEL_MAX_TOKENS.keys()), index=0, key="nome_modelo")
    temperature = st.slider("Nível de Criatividade", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="temperatura")
    interaction_number = len(load_api_usage()) + 1

    fetch_clicked = st.button("Buscar Resposta")
    refine_clicked = st.button("Refinar Resposta")
    evaluate_clicked = st.button("Avaliar Resposta com RAG")
    refresh_clicked = st.button("Apagar")

    references_file = st.file_uploader("Upload do arquivo JSON ou PDF com referências (opcional)", type=["json", "pdf"], key="arquivo_referencias")

with col2:
    container_saida = st.container()

    chat_history = load_chat_history()[-memory_selection:]

    if fetch_clicked:
        if references_file:
            df = upload_and_extract_references(references_file)
            if isinstance(df, pd.DataFrame):
                st.write("### Dados Extraídos do PDF")
                st.dataframe(df)
                st.session_state.references_path = "references.csv"
                st.session_state.references_df = df

        st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente = fetch_assistant_response(user_input, user_prompt, model_name, temperature, agent_selection, chat_history, interaction_number, st.session_state.get('references_df'))
        st.session_state.resposta_original = st.session_state.resposta_assistente
        st.session_state.resposta_refinada = ""
        save_chat_history(user_input, user_prompt, st.session_state.resposta_assistente)

    if refine_clicked:
        if st.session_state.resposta_assistente:
            references_context = ""
            if not st.session_state.references_df.empty:
                for index, row in st.session_state.references_df.iterrows():
                    titulo = row.get('titulo', row['Text'][:50] + '...')
                    autor = row.get('autor', 'Autor Desconhecido')
                    ano = row.get('ano', 'Ano Desconhecido')
                    paginas = row.get('Page', 'Página Desconhecida')
                    references_context += f"Título: {titulo}\nAutor: {autor}\nAno: {ano}\nPágina: {paginas}\n\n"
            st.session_state.resposta_refinada = refine_response(st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, user_input, user_prompt, model_name, temperature, references_context, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.resposta_refinada)
        else:
            st.warning("Por favor, busque uma resposta antes de refinar.")

    if evaluate_clicked:
        if st.session_state.resposta_assistente and st.session_state.descricao_especialista_ideal:
            st.session_state.rag_resposta = evaluate_response_with_rag(user_input, user_prompt, st.session_state.descricao_especialista_ideal, st.session_state.descricao_especialista_ideal, st.session_state.resposta_assistente, model_name, temperature, chat_history, interaction_number)
            save_chat_history(user_input, user_prompt, st.session_state.rag_resposta)
        else:
            st.warning("Por favor, busque uma resposta e forneça uma descrição do especialista antes de avaliar com RAG.")

    with container_saida:
        st.write(f"**#Análise do Especialista:**\n{st.session_state.descricao_especialista_ideal}")
        st.write(f"\n**#Resposta do Especialista:**\n{st.session_state.resposta_original}")
        if st.session_state.resposta_refinada:
            st.write(f"\n**#Resposta Refinada:**\n{st.session_state.resposta_refinada}")
        if st.session_state.rag_resposta:
            st.write(f"\n**#Avaliação com RAG:**\n{st.session_state.rag_resposta}")

    st.markdown("### Histórico do Chat")
    if chat_history:
        tab_titles = [f"Interação {i+1}" for i in range(len(chat_history))]
        tabs = st.tabs(tab_titles)
        
        for i, entry in enumerate(chat_history):
            with tabs[i]:
                st.write(f"**Entrada do Usuário:** {entry['user_input']}")
                st.write(f"**Prompt do Usuário:** {entry['user_prompt']}")
                st.write(f"**Resposta do Especialista:** {entry['expert_response']}")
                st.markdown("---")

if refresh_clicked:
    clear_chat_history()
    st.session_state.clear()
    st.rerun()

st.sidebar.image("logo.png", width=200)
with st.sidebar.expander("Insights do Código"):
    st.markdown("""
    O código do **Consultor de PDFs + IA** é um exemplo de uma aplicação de chat baseada em modelos de linguagem (LLMs) utilizando a biblioteca **Streamlit** e a **API Groq**. Vamos analisar detalhadamente o código, discutir suas inovações, pontos positivos e limitações.

    ### 🧠 **Inovações:**
    - **Suporte a múltiplos modelos de linguagem:** 
        - O código permite a seleção entre diferentes modelos de linguagem, como `Mixtral`, `LLaMA`, e `Gemma`, possibilitando respostas mais precisas e personalizadas.
        - 📌 **Exemplo:** A capacidade de alternar entre `llama3-70b-8192` e `gemma-7b-it` conforme a necessidade da consulta.
    - **Integração com a API Groq:** 
        - Utiliza a capacidade de processamento de linguagem natural de alta performance para gerar respostas precisas.
        - 📌 **Exemplo:** O uso da função `get_completion` que chama a API Groq para gerar respostas com base em prompts fornecidos.
    - **Refinamento de respostas:** 
        - Permite o refinamento das respostas iniciais do modelo de linguagem, tornando-as mais detalhadas e relevantes.
        - 📌 **Exemplo:** A função `refine_response` ajusta a resposta inicial adicionando mais contexto e exemplos específicos.
    - **Avaliação com o RAG (Rational Agent Generator):** 
        - Avalia a qualidade e a precisão das respostas geradas pelo modelo de linguagem.
        - 📌 **Exemplo:** A função `evaluate_response_with_rag` que utiliza critérios como coerência, precisão e relevância para avaliar a resposta.
    - **Manipulação de PDFs:**
        - Capacidade de extrair texto de arquivos PDF, identificar seções e converter em formato de tabela.
        - 📌 **Exemplo:** `extrair_texto_pdf` e `text_to_dataframe` transformam o conteúdo do PDF em um DataFrame do pandas.
    - **Histórico de Chat e Uso de API:**
        - Registra e apresenta o histórico de interações e o uso de APIs, permitindo análises detalhadas.
        - 📌 **Exemplo:** `save_chat_history` e `log_api_usage` armazenam as interações e o uso da API para análise posterior.

    ### 👍 **Pontos positivos:**
    - **Personalização:**
        - Escolha entre diferentes modelos de linguagem e ajuste das respostas conforme as necessidades do usuário.
        - 📌 **Exemplo:** Slider de temperatura para ajustar a criatividade das respostas geradas (`st.slider`).
    - **Precisão:**
        - Integração com a API Groq e refinamento de respostas garantem alta precisão e relevância.
        - 📌 **Exemplo:** Alternância de chaves API para contornar limites de taxa (`handle_rate_limit`).
    - **Flexibilidade:**
        - Suporte a múltiplos modelos e customizações nas respostas.
        - 📌 **Exemplo:** Seleção de modelos através de `st.selectbox`.
    - **Facilidade de Uso:**
        - Interface intuitiva com Streamlit torna a aplicação acessível.
        - 📌 **Exemplo:** `st.text_area` e `st.button` para interações com o usuário.
    - **Gerenciamento de Taxas de API:**
        - Alternância automática entre chaves de API disponíveis.
        - 📌 **Exemplo:** `get_next_api_key` troca as chaves para manter a continuidade do serviço.

    ### ⚠️ **Limitações:**
    - **Dificuldade de uso para iniciantes:**
        - Pode ser desafiador para usuários sem experiência com LLMs ou APIs.
        - **Solução:** Fornecer documentação detalhada e tutoriais.
    - **Limitações de token:**
        - Limitação no número de tokens processados pelo modelo.
        - 📌 **Exemplo:** `MODEL_MAX_TOKENS` define os limites específicos para cada modelo.
    - **Necessidade de treinamento adicional:**
        - Modelos podem precisar de mais treinamento para lidar com consultas complexas.
        - **Solução:** Adaptação de modelos para nichos específicos.
    - **Dependência de APIs externas:**
        - Desempenho e disponibilidade dependem de APIs externas.
        - **Solução:** Alternativas de backup ou redundância.

    ### 🌐 **Importância de Instruções em Chinês:**
    - **Densidade de Informação:**
        - A língua chinesa tem alta densidade de informação, necessitando menos tokens para compreender e gerar respostas.
        - 📌 **Exemplo:** Com menos tokens, um modelo pode processar mais informações em chinês, melhorando a eficiência.
    - **Relevância em Contextos Multilíngues:**
        - Garante que o aplicativo seja eficaz em lidar com consultas em chinês.
        - 📌 **Exemplo:** O prompt em chinês na função `fetch_assistant_response` demonstra a aplicação prática.

    ### 🔍 **Análise Técnica:**
    - **Uso de PDFPlumber para Extração de Texto:**
        - O `pdfplumber` é utilizado para extrair texto de PDFs de maneira eficiente.
        - 📌 **Exemplo:** A função `extrair_texto_pdf` processa cada página do PDF, extraindo e armazenando o texto.
        - **Vantagens:** Permite uma extração precisa do conteúdo textual, mesmo de PDFs complexos.
    - **Conversão para DataFrame:**
        - Conversão de texto extraído para DataFrame facilita análise e manipulação dos dados.
        - 📌 **Exemplo:** `text_to_dataframe` organiza o texto em um formato tabular, utilizando pandas.
        - **Vantagens:** Os DataFrames do pandas oferecem flexibilidade para manipulação de dados, incluindo filtragem, agregação e visualização.
        - **Benefícios:** A estrutura tabular permite operações de análise de dados mais sofisticadas e integração fácil com outras bibliotecas de dados.
    - **Unicode e Suporte Multilíngue:**
        - Suporte a caracteres Unicode para lidar com textos em múltiplos idiomas.
        - 📌 **Exemplo:** `salvar_como_json` usa `ensure_ascii=False` para garantir que caracteres especiais sejam preservados.
        - **Vantagens:** Permite manipulação de textos em diferentes idiomas sem perda de informação, essencial para aplicativos multilíngues.
        - **Benefícios:** Manutenção da integridade dos dados textuais em qualquer idioma, melhorando a precisão das respostas geradas.
    - **Identificação de Seções:**
        - Regex para detectar e organizar seções de texto.
        - 📌 **Exemplo:** A função `identificar_secoes` usa padrões de regex para separar capítulos e partes do texto.
        - **Vantagens:** Organização eficiente do conteúdo textual em seções lógicas, facilitando a navegação e a análise.
        - **Benefícios:** Melhora a estrutura e a clareza do conteúdo extraído, permitindo uma análise mais precisa.
    - **Visualização de Dados:**
        - Utiliza `matplotlib` e `seaborn` para criar gráficos.
        - 📌 **Exemplo:** `plot_api_usage` cria histogramas para visualizar o uso de tokens e tempo por ação de API.
        - **Vantagens:** Ferramentas de visualização robustas que permitem análises visuais detalhadas.
        - **Benefícios:** Gráficos claros e informativos que ajudam a identificar padrões e insights nos dados de uso da API.

    Em resumo, o código é uma aplicação inovadora que combina modelos de linguagem com a API Groq para proporcionar respostas precisas e personalizadas. No entanto, é importante considerar as limitações do aplicativo e trabalhar para melhorá-lo ainda mais. 

    ### 📊 **DataFrames e Unicode:**
    - **Uso de DataFrames:** 
        - DataFrames são utilizados para organizar e manipular grandes volumes de dados de maneira eficiente.
        - 📌 **Exemplo:** A função `text_to_dataframe` transforma texto extraído em um DataFrame, permitindo operações de análise e visualização.
        - **Vantagens:** Flexibilidade para filtrar, agrupar e agregar dados, além de suporte para operações complexas.
        - **Benefícios:** Facilita a integração com outras bibliotecas de análise de dados e visualização.
    - **Suporte a Unicode:**
        - Unicode é essencial para lidar com textos em múltiplos idiomas sem perda de informação.
        - 📌 **Exemplo:** `salvar_como_json` usa `ensure_ascii=False` para preservar caracteres especiais ao salvar dados em JSON.
        - **Vantagens:** Garantia de que os dados textuais sejam preservados corretamente, independentemente do idioma.
        - **Benefícios:** Aumenta a precisão e a integridade dos dados, crucial para aplicações multilíngues.

    """)



    st.sidebar.image("eu.ico", width=80)
    st.sidebar.write("""
    Projeto Consultor de PDFs + IA 
    - Professor: Marcelo Claro.

    Contatos: marceloclaro@gmail.com

    Whatsapp: (88)981587145

    Instagram: [https://www.instagram.com/marceloclaro.geomaker/](https://www.instagram.com/marceloclaro.geomaker/)
    """)

api_usage = load_api_usage()
if api_usage:
    plot_api_usage(api_usage)

if st.sidebar.button("Resetar Gráficos"):
    reset_api_usage()

def carregar_referencias():
    if os.path.exists('references.csv'):
        return pd.read_csv('references.csv')
    else:
        return pd.DataFrame()

def referencias_para_historico(df_referencias, chat_history_file=CHAT_HISTORY_FILE):
    if not df_referencias.empty:
        for _, row in df_referencias.iterrows():
            titulo = row.get('titulo', row['Text'][:50] + '...')
            autor = row.get('autor', 'Autor Desconhecido')
            ano = row.get('ano', 'Ano Desconhecido')
            paginas = row.get('Page', 'Página Desconhecida')
            
            chat_entry = {
                'user_input': f"Título: {titulo}",
                'user_prompt': f"Autor: {autor}\nAno: {ano}\nPágina: {paginas}\nTexto: {row['Text']}",
                'expert_response': 'Informação adicionada ao histórico de chat como referência.'
            }
            
            if os.path.exists(chat_history_file):
                with open(chat_history_file, 'r+') as file:
                    try:
                        chat_history = json.load(file)
                    except json.JSONDecodeError:
                        chat_history = []
                    chat_history.append(chat_entry)
                    file.seek(0)
                    json.dump(chat_history, file, indent=4)
            else:
                with open(chat_history_file, 'w') as file:
                    json.dump([chat_entry], file, indent=4)

df_referencias = carregar_referencias()
referencias_para_historico(df_referencias)
