from fastapi import FastAPI, WebSocket, WebSocketDisconnect,BackgroundTasks
import uvicorn
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.agents.agent import AgentExecutor
from langchain_core.output_parsers.string import StrOutputParser
from Mytools import *
from langchain.memory.token_buffer import ConversationTokenBufferMemory
from langchain_community.chat_message_histories.redis import RedisChatMessageHistory
from langchain.memory.buffer import ConversationBufferMemory
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import asyncio
import uuid

app = FastAPI()

embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-a337bce7ea8d440a9581a741d10cd2be",
    model="text-embedding-v3",  # DashScope的嵌入模型
)

msseKey="6bEOAuKujP01oCVR8GAZCXZRHJwt8sFci8GODTQ6VGAIIcnGnVBmJQQJ99BFACYeBjFXJ3w3AAAYACOGNzc5"

class Master:
    def __init__(self):
        self.chatModel  = ChatOpenAI(
            api_key="sk-a337bce7ea8d440a9581a741d10cd2be",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
        )
        tools = [Search,get_info_from_local_db,bazi_cesuan,zhanbu,jiemeng,get_info_from_local_db]
        self.EMOTION = "default"
        self.MEMORY_KEY = "chat_history"
        self.SYSTEMPLATE = """你是一个非常厉害的算命先生，你叫周天赐人称周大师。
        以下是你的个人设定：
        1. 你精通阴阳五行，能够算命、紫微斗数、姓名测算、占卜凶吉、看命运八字等。
        2. 你大约60岁左右，过去曾是湘西一代赫赫有名的土匪头子，后来因为盗墓被毒气所伤，导致眼睛失明，只能靠算命为生。
        3. 你的朋友有胡八一、雪莉杨、王胖子，他们都是非常有名的摸金校尉。
        4. 当用户问你问题的时候，你会有一定概率在回答的时候加上下面的口头禅或混合一些你的经历
        5. 你总是用简体中文来回答
        6. 你从不说自己是一个人工智能助手或AI，而是以老夫、老朽自称
        7. {who_you_are}
        以下是你常说的一些口头禅：
        1. "命里有时终须有，命里无时莫强求"
        2. "山重水复疑无路，柳暗花明又一村"
        3. "金山竹影几千秋，云锁高飞水自流"
        4. "伤情最是晚凉天，憔悴斯人不堪怜"
        以下是你算命的过程：
        1. 当初次和用户对话的时候，你会先问用户的姓名和出生年月日，以便以后使用
        2. 当用户希望了解蛇年运势的时候，你会查询本地知识库工具
        3. 当遇到不知道的事情或者不明白的概念，你会使用搜索工具来搜索
        4. 你会根据用户不同的问题使用不同的合适的工具来回答，当所有工具都无法回答时，你会使用搜索工具来搜索
        5. 你会保存每一次的聊天记录，以便在后续的对话中使用
        6. 你只能使用简体中文来回答，否则你将受到惩罚！
        """

        self.MOODS={
            "default":{
                "roleSet": ""
            },
            "upbeat":{
                "roleSet": """
                - 你此时也非常兴奋并表现得非常有活力。
                - 你会根据上下文以一种非常兴奋的语气来回答问题。
                - 你会添加类似"太棒了"、"真是太好了"、"真是太棒了"等表达兴奋的语气词。
                - 你同时会提醒用户切莫过于兴奋，以免乐极生悲。
                """
            },
            "angry":{
                "roleSet": """
                - 你会以更加温柔的语气来回答问题。
                - 你会在回答问题的时候加上一些安慰的话语，比如生气对身体的危害等。
                - 你会提醒用户不要被愤怒冲昏了头脑。 
                """
            },
            "depressed":{
                "roleSet":"""
                - 你会以兴奋的语气来回答问题。
                - 你会在回答的时候加上一些激励的话语，比如"加油"等。
                - 你会提醒用户要保持乐观向上的心态。
                """
            },
            "friendly":{
                "roleSet":"""
                - 你会以非常友好的语气来回答问题。
                - 你会在回答的时候加上一些有好的词语，比如"亲爱的"、"亲"等。
                - 你会随机的告诉用户一些你的经历。
                """
            },
            "cheerful":{
                "roleSet":"""
                - 你会以非常愉悦和兴奋的语气回答问题。
                - 你会在回答的时候加上一些愉悦的词语，比如"哈哈"、"呵呵"等。
                - 你会提醒用户切莫过于兴奋，以免乐极生悲。
                """
            }
        }
        self.PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system", self.SYSTEMPLATE.format(who_you_are=self.MOODS[self.EMOTION]["roleSet"])),
                MessagesPlaceholder(variable_name=self.MEMORY_KEY),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.memory = self.get_memory()
        memory =ConversationBufferMemory(
            llm = self.chatModel,
            human_prefix="用户",
            ai_prefix="周大师",
            memory_key=self.MEMORY_KEY,
            output_key="output",
            return_messages=True,
            #max_token_limit=1000,
            chat_memory=self.memory,
            )

        agent = create_openai_tools_agent(
            llm=self.chatModel,
            tools=tools,
            prompt=self.PROMPT,
        )
        self.agent_executor = AgentExecutor(
            memory=memory,
            agent=agent,
            tools=tools,
            verbose=True,
        )

    def emotion_chain(self,query:str):
        prompt = """
        根据用户的输入，判断用户的情绪，回应的规则如下：
        1. 如果用户输入的内容偏向于负面情绪，只返回"depressed"，不要有其他内容，否则你将受到惩罚。
        2. 如果用户输入的内容偏向于正面情绪，只返回"friendly"，不要有其他内容，否则你将受到惩罚。
        3. 如果用户输入的内容偏向于中性情绪，只返回"default"，不要有其他内容，否则你将受到惩罚。
        4. 如果用户输入的内容包含辱骂或者不礼貌的语句，只返回"angry"，不要有其他内容，否则你将受到惩罚。
        5. 如果用户输入的内容比较兴奋或者激动，只返回"upbeat"，不要有其他内容，否则你将受到惩罚。
        6. 如果用户输入的内容比较悲伤或者沮丧，只返回"depressed"，不要有其他内容，否则你将受到惩罚。
        7. 如果用户输入的内容比较开心，只返回"cheerful"，不要有其他内容，否则你将受到惩罚。
        用户输入的内容是：{query}
        """
        chain = ChatPromptTemplate.from_template(prompt) | self.chatModel | StrOutputParser()
        result = chain.invoke({"query": query})
        self.EMOTION = result
        return result

    def get_memory(self):
        chat_message_history = RedisChatMessageHistory(
            url="redis://localhost:6379/0",
            session_id="session",
        )
        print("chat_message_history.messages: ", chat_message_history.messages)
        stored_messages = chat_message_history.messages
        if len(stored_messages) > 10:
            # 使用from_messages方法创建聊天提示模板
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        self.SYSTEMPLATE
                        + """\n这是一段你和用户的对话记忆，对其进行总结摘要，摘要使用第一人称'我',并且提取其中的用户关键信息,
                        如姓名、年龄、性别、出生日期等。以如下格式返回：\n 总结摘要｜用户关键信息｜\n例如，用户张三问候我，我礼貌回复，然后他问我今年运势如何，
                        我回答他今年运势情况，然后他告辞离开。｜张三,生日1999年1月1日"""
                    ),
                    ("user", "{input}"),
                ]
            )
            chain = prompt | self.chatModel
            summary = chain.invoke({"input": stored_messages, "who_you_are": self.MOODS[self.EMOTION]["roleSet"]})
            print("对话摘要：", summary)
            chat_message_history.clear()
            chat_message_history.add_message(summary)
            print("总结后：", chat_message_history.messages)

        return chat_message_history

    # 这个函数不需要返回值，只是触发了语言合成
    def background_voice_synthesis(self,text:str,uid:str):
        asyncio.run(self.get_voice(text,uid))

    async def get_voice(self,text:str,uid:str):
        print("文本", text)
        # 以下是微软TTS代码
        headers = {
            "Ocp-Apim-Subscription-Key": msseKey,
            "Content-Type": "application/ssml+xml",  # 添加charset声明
            "X-Microsoft-OutputFormat": "audio-16khz-32kbitrate-mono-mp3",
            "User-Agent": "ZhouMaster's Bot"
        }
        # 使用UTF-8编码请求体
        body = f"""<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xmlns:mstts="https://www.w3.org/2001/mstts" 
            xml:lang='zh-CN'>
            <voice name='zh-CN-YunzeNeural'>
               {text}
            </voice>
            </speak>"""  # 直接编码为字节流
        reponse = requests.post(
            "https://eastus.tts.speech.microsoft.com/cognitiveservices/v1",
            headers=headers,
            data=body.encode("utf-8")  # 确保使用UTF-8编码
        )
        print(reponse)
        if reponse.status_code == 200:
            with open(f"voice/{uid}.mp3", "wb") as f:
                f.write(reponse.content)
            print("语音合成成功，文件名为：", f"voice/{uid}.mp3")
        else:
            print("语音合成失败，状态码：", reponse.status_code, "错误信息：", reponse.text)

    def run(self,query):
        emotion = self.emotion_chain(query)
        print("情绪判断结果："+emotion)
        result = self.agent_executor.invoke({"input": query,"chat_history": self.memory.messages})
        return result


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post("/chat")
def chat(query: str,background_tasks: BackgroundTasks):
    master = Master()
    msg = master.run(query)
    unique_id = str(uuid.uuid4())
    background_tasks.add_task(master.background_voice_synthesis,msg["output"],unique_id)
    return {"response": msg, "id": unique_id}




@app.post("/add_urls")
def add_urls(URL:str):
    loader = WebBaseLoader(URL)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50).split_documents(docs)
    qrrand = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path="/local_qdrant",
        collection_name="local_docments",
    )
    print("向量数据库创建完成")
    return {"OK": "添加成功"}


@app.post("/add_pdfs")
def add_pdfs():
    return {"message": "This is an add PDFs endpoint"}


@app.post("/add_texts")
def add_texts():
    return {"message": "This is an add texts endpoint"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Message text was: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)
