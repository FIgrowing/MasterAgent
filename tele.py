import json

import requests
import telebot
import urllib.parse
import os
import asyncio

bot = telebot.TeleBot('7968776758:AAGIudqYIKDeZ9RT6WoXKukOweF28pdkNuY')

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, "你好，我是周大师，有什么可以帮助你的?")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    try:
        encoded_message = urllib.parse.quote(message.text)
        reponse = requests.post("http://127.0.0.1:8000/chat?query="+encoded_message)
        if reponse.status_code == 200:
            aisay = json.loads(reponse.text)
            print("AI Response:", aisay)
            if "response" in aisay:
                bot.reply_to(message, aisay["response"]["output"])
                audio_path = f"voice/{aisay['id']}.mp3"
                asyncio.run(check_audio(message, audio_path))
            else:
                bot.reply_to(message, "对不起，我不知道怎么回答你")
    except requests.RequestException as e:
        bot.reply_to(message, "对不起，我不知道怎么回答你")

async def check_audio(message, audio_path):
    while True:
        if os.path.exists(audio_path):
            with open(audio_path, 'rb') as audio_file:
                bot.send_audio(message.chat.id, audio_file)
            os.remove(audio_path)
            break
        else:
            print(f"等待音频文件 {audio_path} 生成...")
            # 等待1秒后再次检查
            await asyncio.sleep(1)


bot.infinity_polling()


