import os
import time
import json
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram.ext import Filters, MessageHandler
import numpy as np
import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForQuestionAnswering


MAX_PARAGRAPHS = 100
MAX_ANSWERS = 7
LOGITS_QUANTILE = 0.95
TRANSFORMER_MODEL = "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru"


def load_constitution(path='constitution_rf.json', max_paragraphs=None):
    locations, paragraphs = [], []
    with open(path) as f:
        lines = json.load(f)
    if max_paragraphs is None:
        max_paragraphs = len(lines)
    for line in lines[:max_paragraphs]:
        loc_parts, par_parts = [], []
        for level0, level1 in line.items():
            if '№' in level1:
                loc_parts.append(f"{level0} {level1['№']}")
            else:
                loc_parts.append(level0)
            if 'Текст' in level1:
                par_parts.append(level1['Текст'])
        locations.append(', '.join(loc_parts))
        paragraphs.append('\n'.join(par_parts))
    return locations, paragraphs


def start(update: Update, context: CallbackContext):
    info_text = (
        "Здравствуте, это макет учебного бота, "
        "который учится отвечать на вопросы по (части) Конституции РФ.\n"
        "Задавайте вопросы в чате."
    )
    context.bot.send_message(chat_id=update.effective_chat.id, text=info_text)


def unknown(update: Update, context: CallbackContext):
    reply_text = (
        f'Вы задали команду "{update.message.text}", но бот такую не знает.'
    )
    context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)


class QAMaster:
    def __init__(self, tokenizer, model, paragraphs, locations):
        self.tokenizer = tokenizer
        self.model = model
        self.paragraphs = paragraphs
        self.locations = locations
        self.bos_id = model.config.bos_token_id
        self.eos_id = model.config.eos_token_id
        self.max_emb = model.config.max_position_embeddings

    @staticmethod
    def send_message(update, context, text):
        if os.getenv('BOT_DEBUG_QUESTION'):
            print(text)
        else:
            context.bot.send_message(chat_id=update.effective_chat.id, text=text)

    def answer(self, update: Update, context: CallbackContext):
        intro_answer = "Подождите немного, пожалуйста, я очень медленный."
        self.send_message(update, context, intro_answer)
        try:
            question = update.message.text
        except AttributeError:
            question = os.getenv('BOT_DEBUG_QUESTION', '?')
        answers, logits = [], []
        last_time = time.time()
        for text in self.paragraphs:
            if time.time() - last_time > 15:
                last_time = time.time()
                self.send_message(update, context, "Еще думаю...")
            inputs = tokenizer.encode_plus(question, text,
                                           add_special_tokens=True,
                                           max_length=self.max_emb,
                                           truncation=True,
                                           return_tensors="pt")
            input_ids = inputs["input_ids"][0]
            outputs = model(**inputs)
            answer_start = torch.argmax(outputs.start_logits)
            answer_end = torch.argmax(outputs.end_logits) + 1
            if (
                    answer_start > answer_end or
                    answer_start < torch.nonzero(input_ids == self.eos_id)[0]
            ):
                answers.append('')
                logits.append(float('-Inf'))
                continue
            tokens = tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
            answers.append(tokenizer.convert_tokens_to_string(tokens))
            start_end_logits = torch.vstack([outputs.start_logits, outputs.end_logits])
            logit_factor = start_end_logits.max(dim=1).values.mean()
            logits.append(float(logit_factor))
        logits_order = np.argsort(logits)
        logits = np.array(logits)[logits_order]
        answers = np.array(answers)[logits_order]
        locations = np.array(self.locations)[logits_order]
        idx = np.flatnonzero(logits > np.quantile(logits, LOGITS_QUANTILE))[::-1][:MAX_ANSWERS]
        if idx.size:
            all_answers = '\n'.join(f'"{ans}" ({loc})' for ans, loc in
                                    zip(answers[idx], locations[idx]))
            self.send_message(update, context, f'Возможные значения:\n{all_answers}')
        else:
            self.send_message(update, context, 'Увы, ничего не нашел.')


if __name__ == '__main__':
    # Model
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(TRANSFORMER_MODEL)
    model = XLMRobertaForQuestionAnswering.from_pretrained(TRANSFORMER_MODEL)
    # Paragraphs of the Constitution
    locations, paragraphs = load_constitution(max_paragraphs=MAX_PARAGRAPHS)
    qa = QAMaster(tokenizer, model, paragraphs, locations)
    # Bot
    load_dotenv()
    api_key = os.getenv('TELEGRAM_HTTP_API')
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    if os.getenv('BOT_DEBUG_QUESTION'):
        qa.answer(None, None)
    else:
        updater = Updater(api_key)
        updater.dispatcher.add_handler(CommandHandler('start', start))
        updater.dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, qa.answer))
        updater.dispatcher.add_handler(MessageHandler(Filters.command, unknown))
        # Start the Bot
        updater.start_polling()
        updater.idle()
