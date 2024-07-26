# Consultor de PDFs + IA

## 📄 Visão Geral
Este projeto é um aplicativo de consulta de PDFs com integração de Inteligência Artificial, desenvolvido em Python utilizando Streamlit, PDFPlumber e a API Groq. Ele permite a extração de texto de PDFs, processamento e organização desses dados, e a geração de respostas refinadas por meio de modelos de linguagem avançados. Através deste README, você encontrará uma explicação detalhada das funcionalidades, insights inovadores, e instruções de uso.

## 🚀 Funcionalidades Principais
- **Extração de Texto de PDFs**: Utiliza PDFPlumber para extrair texto de arquivos PDF carregados pelo usuário.
- **Processamento de Dados**: Organiza o texto extraído em um formato estruturado e o transforma em DataFrames do Pandas.
- **Integração com Modelos de Linguagem**: Usa a API Groq para gerar respostas detalhadas baseadas nos dados extraídos.
- **Refinamento de Respostas**: Permite ao usuário refinar respostas iniciais para obter informações mais precisas e relevantes.
- **Avaliação de Respostas**: Avalia a qualidade das respostas usando o Rational Agent Generator (RAG).

## 🛠️ Tecnologias Utilizadas
- **Streamlit**: Framework para construção de aplicações web interativas em Python.
- **PDFPlumber**: Biblioteca para extração de texto de arquivos PDF.
- **Pandas**: Biblioteca para manipulação e análise de dados.
- **Groq API**: Plataforma para integração com modelos de linguagem avançados.

## 🎯 Execução do Projeto
1. **Instalação de Dependências**: Execute `pip install -r requirements.txt` para instalar todas as bibliotecas necessárias.
2. **Execução do Aplicativo**: Inicie o aplicativo com o comando `streamlit run app.py`.
3. **Upload de PDF**: Carregue um arquivo PDF através da interface do usuário para extração e processamento de texto.

## 📊 Detalhamento do Código
### Configuração da Página do Streamlit
O código configura a página inicial do Streamlit com título, ícone e layout. Utiliza `st.set_page_config` para definir essas propriedades, garantindo uma interface amigável e personalizada para o usuário.

### Definição de Constantes e Chaves de API
Constantes como caminhos de arquivos e tokens máximos dos modelos são definidos no início do código. Além disso, chaves de API para diferentes ações (`fetch`, `refine`, `evaluate`) são armazenadas em um dicionário, permitindo alternância entre elas para evitar limites de taxa.

### Manipulação de Limites de Taxa
Funções como `handle_rate_limit` são implementadas para gerenciar limites de taxa impostos pela API. Quando uma taxa é atingida, a função espera por um período especificado antes de tentar novamente, garantindo continuidade no processamento.

### Extração e Processamento de Texto de PDFs
Funções como `extrair_texto_pdf` utilizam PDFPlumber para ler e extrair texto de cada página de um PDF. O texto extraído é então estruturado em uma lista de dicionários, com cada dicionário representando uma página.

### Identificação de Seções em Texto
A função `identificar_secoes` organiza o texto extraído em seções, identificando cabeçalhos baseados em padrões específicos. Isso é essencial para uma análise estruturada e compreensível dos dados.

### Salvamento de Dados
Os dados processados podem ser salvos em formato JSON usando a função `salvar_como_json`. Isso facilita a persistência e reutilização das informações extraídas e organizadas.

### Interação com a API Groq
Funções como `fetch_assistant_response` interagem com a API Groq para obter respostas baseadas em prompts fornecidos pelo usuário. A API usa modelos de linguagem avançados para gerar respostas detalhadas e contextualmente relevantes.

### Refinamento e Avaliação de Respostas
Após obter uma resposta inicial, o usuário pode refiná-la para obter mais precisão. Além disso, o código permite a avaliação da resposta usando o RAG, proporcionando uma análise crítica e melhorias contínuas nas respostas.

### Visualização de Dados
Gráficos de uso da API são gerados usando Matplotlib e Seaborn, permitindo ao usuário visualizar o desempenho e o consumo de recursos das chamadas de API. Isso inclui histogramas de uso de tokens e tempo por chamada de API.

### Histórico e Uso de API
Funções para salvar e carregar histórico de chat e uso da API são implementadas, garantindo que todas as interações sejam registradas e possam ser revisadas posteriormente. Isso é crucial para manter um registro completo das atividades e interações do usuário.

## 💡 Insights Inovadores
- **Troca Dinâmica de Chaves de API**: A capacidade de alternar entre várias chaves de API permite evitar interrupções devido a limites de taxa, garantindo uma experiência contínua para o usuário.
- **Processamento Estruturado de Texto**: A organização do texto extraído em seções facilita a análise e a geração de respostas precisas, mostrando um avanço em comparação com métodos não estruturados.
- **Refinamento Iterativo de Respostas**: A possibilidade de refinar respostas várias vezes permite obter informações extremamente precisas e relevantes, adaptando-se continuamente às necessidades do usuário.
- **Avaliação com RAG**: O uso do RAG para avaliar respostas garante que as informações fornecidas sejam de alta qualidade e relevância, incorporando um nível de análise crítica que aumenta a confiança nas respostas geradas.
- **Flexibilidade e Personalização**: A capacidade de escolher entre diferentes modelos de linguagem e ajustar a criatividade das respostas proporciona uma personalização avançada, atendendo a diversas preferências e necessidades dos usuários.

## 📝 Exemplo de Uso
Para ilustrar o funcionamento do aplicativo, suponha que um usuário deseja extrair texto de um PDF sobre inteligência artificial, organizar esse texto em seções e obter uma resposta detalhada sobre um tópico específico. O usuário carrega o PDF, processa o texto, e então faz uma consulta ao modelo de linguagem. Após obter a resposta inicial, ele pode refiná-la e, se necessário, avaliar sua qualidade usando o RAG.

## 🗃️ Estrutura do Projeto
```
Consultor_de_PDFs_IA/
├── app.py
├── requirements.txt
├── agents.json
├── chat_history.json
├── api_usage.json
└── README.md
```

## 🤝 Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir issues e pull requests para melhorias e novas funcionalidades.

## 📞 Contato
Para mais informações, dúvidas ou sugestões, entre em contato com o desenvolvedor:
- **Email**: marceloclaro@gmail.com
- **Whatsapp**: (88) 98158-7145
- **Instagram**: [@marceloclaro.consultorpdfs](https://www.instagram.com/marceloclaro.consultorpdfs/)

## 📢 Licença
Este projeto é licenciado sob os termos da licença MIT. Para mais detalhes, consulte o arquivo LICENSE.

---
