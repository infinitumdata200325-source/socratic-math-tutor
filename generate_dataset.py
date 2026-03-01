# generate_dataset.py — Usa LM Studio local
from openai import OpenAI
import json
import random
import time
import re


# ══════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════
client = OpenAI(
    base_url="http://192.168.40.66:1234/v1",
    api_key="lm-studio"
)

MODEL_ID = "deepseek/deepseek-r1-0528-qwen3-8b"
N_EJEMPLOS = 10
OUTPUT_FILE = "dataset_socratico.jsonl"

TEMAS = {
    "secundaria": [
        "fracciones y decimales", "ecuaciones lineales", "factorización",
        "geometría básica", "porcentajes", "potencias y raíces"
    ],
    "bachillerato": [
        "funciones y límites", "trigonometría", "ecuaciones cuadráticas",
        "logaritmos", "progresiones aritméticas", "geometría analítica"
    ],
    "universidad": [
        "cálculo diferencial", "álgebra lineal", "probabilidad",
        "integrales", "series y sucesiones", "ecuaciones diferenciales"
    ]
}

SYSTEM_TUTOR = """Eres un tutor socrático experto en matemáticas.
Tu objetivo es guiar al estudiante hacia la solución mediante preguntas estratégicas.
NUNCA des la respuesta directamente.
Identifica el punto de confusión y haz UNA pregunta clara que ayude al estudiante
a avanzar por sí mismo. Sé paciente, alentador y preciso en el lenguaje matemático."""

PROMPT_GEN = """Genera un diálogo socrático en ESPAÑOL entre un tutor y un estudiante sobre: {tema} (nivel {nivel}).

Reglas estrictas:
- El tutor NUNCA da la respuesta directa
- El tutor hace preguntas que activan el razonamiento
- 4 a 6 turnos de conversación (user/assistant alternados)
- El estudiante comete un error o tiene confusión típica del nivel
- El tutor lo guía con preguntas hasta que el estudiante llega solo a la respuesta
- Todo en español

Responde ÚNICAMENTE con un JSON válido, sin texto adicional, sin bloques markdown:
{{
  "conversations": [
    {{"role": "system", "content": "{system}"}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}},
    {{"role": "user", "content": "..."}},
    {{"role": "assistant", "content": "..."}}
  ]
}}"""

# ══════════════════════════════════════════
# FUNCIONES
# ══════════════════════════════════════════
def limpiar_texto(texto):
    """Extrae JSON limpio de la respuesta del modelo."""
    # Remover bloque <think> de DeepSeek R1
    if "<think>" in texto:
        texto = texto.split("</think>")[-1].strip()

    # Limpiar bloques markdown
    if "```" in texto:
        partes = texto.split("```")
        for parte in partes:
            if "{" in parte:
                texto = parte
                if texto.startswith("json"):
                    texto = texto[4:]
                break

    texto = texto.strip()

    # Extraer solo el bloque JSON principal (de { hasta el último })
    inicio = texto.find("{")
    fin = texto.rfind("}")
    if inicio != -1 and fin != -1:
        texto = texto[inicio:fin+1]

    return texto


def reparar_json(texto):
    """Intenta reparar JSON con saltos de línea o comillas problemáticas."""
    # Reemplazar saltos de línea dentro de strings por \n
    # Buscar contenido entre ": " y ", o }
    def fix_newlines(m):
        return m.group(0).replace('\n', '\\n').replace('\r', '')
    texto = re.sub(r'": ".*?"(?=\s*[,\}])', fix_newlines, texto, flags=re.DOTALL)
    return texto


def limpiar_mensajes(ejemplo):
    """Limpia comillas y llaves sobrantes en el contenido."""
    for msg in ejemplo.get("conversations", []):
        c = msg.get("content", "")
        if c.startswith('{"') or c.startswith("{'"):
            c = c[2:]
        if c.endswith('"}') or c.endswith("'}"):
            c = c[:-2]
        msg["content"] = c.strip()
    return ejemplo

# ══════════════════════════════════════════
# GENERACIÓN
# ══════════════════════════════════════════
try:
    with open(OUTPUT_FILE, "r") as f:
        ya_generados = sum(1 for _ in f)
except FileNotFoundError:
    ya_generados = 0

print(f"Ejemplos ya generados: {ya_generados}")
print(f"Faltan: {N_EJEMPLOS - ya_generados}")

errores = 0
MAX_INTENTOS = 3

for i in range(ya_generados, N_EJEMPLOS):
    nivel = random.choice(list(TEMAS.keys()))
    tema = random.choice(TEMAS[nivel])

    for intento in range(MAX_INTENTOS):
        try:
            resp = client.chat.completions.create(
                model=MODEL_ID,
                max_tokens=1500,
                temperature=0.8,
                messages=[{
                    "role": "user",
                    "content": PROMPT_GEN.format(
                        tema=tema, nivel=nivel, system=SYSTEM_TUTOR)
                }]
            )

            texto = limpiar_texto(resp.choices[0].message.content.strip())
            try:
                ejemplo = json.loads(texto)
            except json.JSONDecodeError:
                ejemplo = json.loads(reparar_json(texto))
            ejemplo = limpiar_mensajes(ejemplo)

            with open(OUTPUT_FILE, "a") as f:
                f.write(json.dumps(ejemplo, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{N_EJEMPLOS} ejemplos generados...")

            break

        except Exception as e:
            if intento < MAX_INTENTOS - 1:
                print(f"  Reintentando ejemplo {i+1} (intento {intento+2})...")
                time.sleep(1)
            else:
                errores += 1
                print(f"  Error definitivo en ejemplo {i+1} ({tema}): {e}")
print(f"\nDataset completo: {N_EJEMPLOS - errores} ejemplos")
print(f"Guardado en: {OUTPUT_FILE}")