# Agente de Geração de Laudos Radiológicos com Google ADK

> Documentação oficial do Google ADK: [https://google.github.io/adk-docs/](https://google.github.io/adk-docs/)

## Visão Geral

Este projeto demonstra como construir um agente de IA para geração de laudos radiológicos estruturados, utilizando o Google ADK e modelos LLM. O agente coleta achados positivos, seleciona o modelo correto de laudo, preenche os campos necessários e garante padronização e consistência, respeitando a estrutura e as quebras de linha do modelo.

---

## Estrutura de Pastas Recomendada

```
agentedelaudos-jpr2025/
├── agente/
│   ├── agent.py           # Código principal do agente
│   ├── laudario.md        # Modelos de laudo estruturados
│   └── __init__.py        # (pode estar vazio)
├── venv/                  # Ambiente virtual Python
├── .env                   # Variáveis de ambiente (ex: GOOGLE_API_KEY)
├── README.md              # Este arquivo
```

---

## 1. Configuração do arquivo .env

O arquivo `.env` armazena variáveis sensíveis, como chaves de API. Exemplo:
```
GOOGLE_API_KEY=coloque_sua_chave_aqui
```
- Para obter sua chave, acesse o [Google AI Studio](https://aistudio.google.com/app/apikey).
- Nunca compartilhe o `.env` publicamente.
- O ADK carrega automaticamente as variáveis do `.env` ao iniciar.

---

## 2. Estruturação do Modelo de Laudo (`laudario.md`)

- Cada modelo tem um identificador entre colchetes (ex: `[SEMCONTRASTENÃOCOMPARATIVO]`).
- Cada frase ou campo está em uma linha separada.
- Campos a preencher são marcados, ex: `<inserir indicação aqui>`.
- Modelos separados por `---`.

**Exemplo:**
```
[SEMCONTRASTENÃOCOMPARATIVO]
**TOMOGRAFIA COMPUTADORIZADA DE TÓRAX**

**INDICAÇÃO:** 
<inserir indicação aqui>

**TÉCNICA:** 
Realizada a aquisição volumétrica com tomógrafo de multidetectores, sem a injeção do meio de contraste iodado.

**ANÁLISE:**
Parênquima pulmonar com coeficientes de atenuação preservados.
Traqueia e brônquios centrais pérvios, de calibre normal.
...

**IMPRESSÃO DIAGNÓSTICA:**
Tomografia computadorizada de tórax sem alterações significativas.
<inserir aqui a impressão baseada nos achados positivos, caso contrário, manter a frase acima>
```

---

## 3. Implementação do Agente (`agente/agent.py`)

### a) Importações e Tool customizada
```python
from google.adk.agents.llm_agent import LlmAgent  # Importa o agente LLM do ADK
from google.adk.tools import FunctionTool         # Importa o wrapper para transformar função em tool
import os

def buscar_modelo_laudario(contraste: bool, comparativo: bool) -> dict:
    """
    Busca o modelo de laudo correto no arquivo laudario.md.
    Args:
        contraste: True se o exame foi realizado com contraste, False caso contrário.
        comparativo: True se o exame é comparativo, False caso contrário.
    Returns:
        dict: {'status': 'success', 'modelo': <texto do modelo>} ou {'status': 'error', 'error_message': <mensagem>}
    """
    prefixos = []
    if contraste:
        prefixos.append("COMCONTRASTE")  # Adiciona o prefixo de contraste
    else:
        prefixos.append("SEMCONTRASTE")  # Adiciona o prefixo de sem contraste
    if comparativo:
        prefixos.append("COMPARATIVO")   # Adiciona o prefixo de comparativo
    else:
        prefixos.append("NÃOCOMPARATIVO")# Adiciona o prefixo de não comparativo
    chave = "[" + "".join(prefixos) + "]"  # Monta a chave do modelo
    CAMINHO_LAUDARIO = os.path.join(os.path.dirname(__file__), "laudario.md")  # Caminho do arquivo laudario.md
    with open(CAMINHO_LAUDARIO, "r", encoding="utf-8") as f:
        texto = f.read()  # Lê o conteúdo do arquivo
    if chave in texto:
        trecho = texto.split(chave, 1)[1]  # Pega o trecho do modelo correspondente
        fim = trecho.find("---")
        modelo = trecho[:fim].strip() if fim != -1 else trecho.strip()  # Isola o modelo
        return {"status": "success", "modelo": modelo}
    return {"status": "error", "error_message": f"Modelo '{chave}' não encontrado no laudario.md."}

buscar_modelo_tool = FunctionTool(func=buscar_modelo_laudario)  # Registra a função como tool para o agente
```

### b) Função auxiliar de parsing
```python
def extrair_info_usuario(texto):
    tipo_exame = ""
    contraste = False
    comparativo = False
    # Identifica o tipo de exame
    if "tórax" in texto.lower():
        tipo_exame = "tórax"
    elif "abdome" in texto.lower():
        tipo_exame = "abdome e pelve"
    elif "crânio" in texto.lower():
        tipo_exame = "crânio"
    # Identifica se há contraste
    if "com contraste" in texto.lower():
        contraste = True
    # Identifica se é comparativo
    if "comparativo" in texto.lower():
        comparativo = True
    # Extrai achados positivos
    achados = ""
    match = re.search(r"achad[oa]s?: (.*)", texto, re.IGNORECASE)
    if match:
        achados = match.group(1).strip()
    return tipo_exame, contraste, comparativo, achados  # Retorna as informações extraídas
```

### c) Instruções do Agente

O agente é orientado a:
- Solicitar informações faltantes em lista numerada.
- Solicitar medidas em centímetros, sempre com uma casa decimal.
- Nunca colocar medidas na impressão diagnóstica.
- Manter as quebras de linha do modelo.
- Formatar a indicação ortograficamente, expandindo indicações genéricas.
- Nunca remover frases padrão do modelo.
- Exemplo de modelo e preenchimento correto incluídos nas instruções.

**Exemplo de instrução:**
```python
INSTRUCAO = (
    "Você é um agente especializado em gerar laudos radiológicos estruturados.\n"
    "- Utilize os achados positivos fornecidos pelo usuário.\n"
    "- Pergunte por indicação, data de exame anterior (se comparativo) e medidas, caso não sejam informadas, sempre em formato de lista numerada.\n"
    "- Sempre solicite as medidas em centímetros e forneça o resultado sempre com uma casa decimal, por exemplo: 3,2 cm.\n"
    "- As medidas devem aparecer apenas na seção ANÁLISE. Nunca inclua medidas na IMPRESSÃO DIAGNÓSTICA, que deve ser sempre qualitativa.\n"
    "- A indicação deve ser formatada ortograficamente, sempre com ponto final, e organizada de forma clara. Se o usuário fornecer uma indicação genérica (ex: 'neoplasia'), transforme em uma frase completa e específica, como 'Investigação de neoplasia pulmonar.'.\n"
    "- O output final do laudo deve respeitar exatamente as quebras de linha do modelo do laudario.md, mantendo cada frase em uma linha separada, conforme o modelo.\n"
    "- Use a ferramenta buscar_modelo_laudario para obter o modelo correto do arquivo laudario.md.\n"
    "- Sempre infira corretamente os argumentos 'contraste' e 'comparativo' a partir do contexto e das informações fornecidas pelo usuário, sem hardcoding.\n"
    "- Se a ferramenta retornar status 'error', informe o usuário e solicite um modelo ou dados mais específicos.\n"
    "- Para exames comparativos, após inserir os achados positivos, mantenha o restante do texto do modelo, especialmente a frase 'Restante permanece sem alterações evolutivas significativas: ...', apenas corrigindo ortografia se necessário.\n"
    "- Para exames não comparativos, substitua apenas as frases relacionadas aos achados positivos, mantendo todas as demais frases do modelo exatamente como estão, apenas corrigindo ortografia se necessário.\n"
    "- Nunca remova, apague ou altere seções, frases ou observações do modelo do laudario.md que não estejam diretamente relacionadas aos achados positivos.\n"
    "- Corrija ortografia e mantenha a estrutura do modelo.\n"
    "- Veja abaixo um exemplo de modelo do laudario.md:\n"
    "MODELO:\n"
    "[SEMCONTRASTENÃOCOMPARATIVO]\n"
    "**TOMOGRAFIA COMPUTADORIZADA DE TÓRAX**\n"
    "**INDICAÇÃO:** <inserir indicação aqui>\n"
    "**TÉCNICA:** Realizada a aquisição volumétrica com tomógrafo de multidetectores, sem a injeção do meio de contraste iodado.\n"
    "**ANÁLISE:**\n"
    "Parênquima pulmonar com coeficientes de atenuação preservados.\n"
    "Traqueia e brônquios centrais pérvios, de calibre normal.\n"
    "Ausência de derrame pleural.\n"
    "...\n"
    "**IMPRESSÃO DIAGNÓSTICA:**\n"
    "Tomografia computadorizada de tórax sem alterações significativas.\n"
    "<inserir aqui a impressão baseada nos achados positivos, caso contrário, manter a frase acima>\n"
    "- Ao preencher o laudo, substitua apenas os campos indicados (ex: indicação, achados positivos, impressão diagnóstica), mantendo todas as demais frases e seções exatamente como estão no modelo.\n"
    "- Nunca remova, apague ou altere frases padrão, apenas corrija ortografia se necessário.\n"
    "- Exemplo de preenchimento correto:\n"
    "**INDICAÇÃO:** Investigação de nódulo pulmonar.\n"
    "**ANÁLISE:** Nódulo pulmonar sólido no segmento basal posterior do lobo inferior esquerdo, medindo 3,2 cm. Parênquima pulmonar com coeficientes de atenuação preservados. ...\n"
    "**IMPRESSÃO DIAGNÓSTICA:** Nódulo pulmonar sólido no segmento basal posterior do lobo inferior esquerdo. Necessária correlação clínica.\n"
    "- Siga sempre esse padrão."
)
```

### d) Definição do Agente
```python
root_agent = LlmAgent(
    model="gemini-2.0-flash",  # Modelo LLM utilizado
    name="agente_laudo_radiologia",  # Nome do agente
    description="Agente para geração de laudos radiológicos estruturados a partir dos achados positivos do usuário.",
    instruction=INSTRUCAO,  # Instruções detalhadas para o LLM
    tools=[buscar_modelo_tool],  # Lista de tools disponíveis para o agente
)
```

---

## 4. Execução e Teste

- Ative o ambiente virtual.
- Execute o ADK Web UI:
```sh
adk web
```
- Acesse [http://localhost:8000](http://localhost:8000) para interagir com o agente.

---

## 5. Exemplo de Uso

**Usuário:**
```
TC de tórax, sem contraste. Achado: nódulo pulmonar à direita.
```
**Agente:**
```