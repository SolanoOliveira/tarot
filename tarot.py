import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# ⚠️ Pegue o token de variável de ambiente (definida no Render ou localmente)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")

# Caminho do adaptador LoRA
ADAPTER_PATH = "./adaptador"

# Carregar modelo base + adaptador LoRA
config = PeftConfig.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Função para gerar resposta
def responder(pergunta):
    prompt = f"### Instrução:\n{pergunta}\n\n### Entrada:\nO consulente escolheu 5 cartas do tarot.\n\n### Resposta:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=150)
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resposta.split("### Resposta:")[-1].strip()

# Lidar com mensagens recebidas
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text
    resposta = responder(pergunta)
    await update.message.reply_text(resposta)

# Inicializar e rodar bot
def main():
    if not TELEGRAM_TOKEN:
        print("❌ TELEGRAM_TOKEN não definido. Defina como variável de ambiente.")
        return

    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot do Tarot rodando...")
    app.run_polling()

if __name__ == "__main__":
    main()
