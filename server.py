from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.agents.agent import AgentExecutor
from langchain.agents import tool
from langchain_core.output_parsers.string import StrOutputParser

app = FastAPI()
@tool(description = "测试")
def test():
    return "test"


class Master:
    def __init__(self):
        self.chatModel = chatLLM = ChatOpenAI(
            api_key="sk-a337bce7ea8d440a9581a741d10cd2be",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
        )
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
        2. 当用户希望了解今年运势的时候，你会查询本地知识库工具
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
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        self.memory = ""
        tools = [test]
        agent = create_openai_tools_agent(
            llm=self.chatModel,
            tools=tools,
            prompt=self.PROMPT,
        )
        self.agent_executor = AgentExecutor(
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

    def run(self,query):
        emotion = self.emotion_chain(query)
        print("情绪判断结果："+emotion)
        result = self.agent_executor.invoke({"input": query})
        return result


@app.get("/")
def read_root():
    return {"hello": "world"}


@app.post("/chat")
def chat(query: str):
    master = Master()
    return master.run(query)


@app.post("/add_url")
def add_url():
    return {"message": "This is an add URL endpoint"}


@app.post("/add_urls")
def add_urls():
    return {"message": "This is an add URLs endpoint"}


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
