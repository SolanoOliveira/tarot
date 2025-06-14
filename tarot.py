from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, ContextTypes, filters

# ⚠️ Substitua pelo seu token do BotFather
TELEGRAM_TOKEN = "7535144647:AAHcF5BC0f4Sqq6IFQClB7wU9eRb6wcTIEE"

# ⚠️ Caminho onde você salvou o adaptador
ADAPTER_PATH = "./adaptador"

# Carregar config para saber qual era o modelo base
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

# Função assíncrona para lidar com mensagens
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pergunta = update.message.text
    resposta = responder(pergunta)
    await update.message.reply_text(resposta)

# Inicializar o bot com a nova estrutura
def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("🤖 Bot rodando... Envie uma mensagem no Telegram!")
    app.run_polling()

if __name__ == "__main__":
    main()
