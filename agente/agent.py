from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
import re
import os

# Tool para buscar o modelo correto no laudario.md
def buscar_modelo_laudario(contraste: bool, comparativo: bool) -> dict:
    """
    Busca o modelo de laudo correto no arquivo laudario.md.
    Args:
        contraste: True se o exame foi realizado com contraste, False caso contrário. A LLM deve inferir esse valor a partir da descrição do exame fornecida pelo usuário.
        comparativo: True se o exame é comparativo, False caso contrário. A LLM deve inferir esse valor a partir do contexto do exame e das informações fornecidas pelo usuário.
    Returns:
        dict: {'status': 'success', 'modelo': <texto do modelo>} ou {'status': 'error', 'error_message': <mensagem>}
    """
    prefixos = []
    if contraste:
        prefixos.append("COMCONTRASTE")
    else:
        prefixos.append("SEMCONTRASTE")
    if comparativo:
        prefixos.append("COMPARATIVO")
    else:
        prefixos.append("NÃOCOMPARATIVO")
    chave = "[" + "".join(prefixos) + "]"
    CAMINHO_LAUDARIO = os.path.join(os.path.dirname(__file__), "laudario.md")
    with open(CAMINHO_LAUDARIO, "r", encoding="utf-8") as f:
        texto = f.read()
    if chave in texto:
        trecho = texto.split(chave, 1)[1]
        fim = trecho.find("---")
        modelo = trecho[:fim].strip() if fim != -1 else trecho.strip()
        return {"status": "success", "modelo": modelo}
    return {"status": "error", "error_message": f"Modelo '{chave}' não encontrado no laudario.md."}

buscar_modelo_tool = FunctionTool(func=buscar_modelo_laudario)

def extrair_info_usuario(texto):
    tipo_exame = ""
    contraste = False
    comparativo = False
    if "tórax" in texto.lower():
        tipo_exame = "tórax"
    elif "abdome" in texto.lower():
        tipo_exame = "abdome e pelve"
    elif "crânio" in texto.lower():
        tipo_exame = "crânio"
    if "com contraste" in texto.lower():
        contraste = True
    if "comparativo" in texto.lower():
        comparativo = True
    achados = ""
    match = re.search(r"achad[oa]s?: (.*)", texto, re.IGNORECASE)
    if match:
        achados = match.group(1).strip()
    return tipo_exame, contraste, comparativo, achados

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

root_agent = LlmAgent(
    model="gemini-2.0-flash",
    name="agente_laudo_radiologia",
    description="Agente para geração de laudos radiológicos estruturados a partir dos achados positivos do usuário.",
    instruction=INSTRUCAO,
    tools=[buscar_modelo_tool],
)