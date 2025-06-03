from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain.agents.openai_tools.base import create_openai_tools_agent
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.agents.agent import AgentExecutor
from langchain.agents import tool

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
        self.MEMORY_KEY = "chat_history"
        self.TEMPLATE = ""
        self.PROMPT = ChatPromptTemplate.from_messages(
            [
                ("system","你是一个乐于助人的AI助手"),
                ("user","{input}"),
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

    def run(self,query):
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
