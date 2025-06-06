import json

from langchain.agents import tool
from langchain_community.utilities.serpapi import SerpAPIWrapper
from langchain_community.vectorstores.qdrant import Qdrant
from qdrant_client import QdrantClient
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers.json import JsonOutputParser
import requests
from langchain_core.prompts import PromptTemplate


YUANFENJU_API_KEY = "YHPRGuiURJ5IzBX6IzqYJBUp8"

embeddings = DashScopeEmbeddings(
    dashscope_api_key="sk-a337bce7ea8d440a9581a741d10cd2be",
    model="text-embedding-v3",  # DashScope的嵌入模型
)

chatLLM = ChatOpenAI(
    api_key="sk-a337bce7ea8d440a9581a741d10cd2be",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    temperature=0,  # 设置温度为0，确保回答更准确
)

SERPAPI_API_KEY = "5b708ee44c3408197d8454ae7861499db2fe25b34c7814b05ef3d98cc474de68"


@tool(description="测试")
def test():
    return "test"


@tool(description="只有需要了解实时信息或者不知道的事情的时候才使用这个工具")
def Search(query: str):
    """
    使用SerpAPIWrapper来搜索实时信息。
    只在需要了解实时信息或者不知道的事情的时候使用。
    """
    search = SerpAPIWrapper(serpapi_api_key=SERPAPI_API_KEY)
    result = search.run(query)
    return result


@tool(description="只回答2025年运势或者蛇年运势相关的问题的时候，会使用这个工具")
def get_info_from_local_db(query: str):
    client = Qdrant(
        QdrantClient(path="/local_qdrant"),
        collection_name="local_docments",
        embedding=embeddings, )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result


@tool(
    description="只有做八字测算的时候才会使用到这个工具，需要输入用户姓名和出生年月日时，如果缺少姓名或者出生年月日时则不可用")
def bazi_cesuan(query: str):
    url = f"https://api.yuanfenju.com/index.php/v1/Bazi/cesuan"
    prompt = ChatPromptTemplate.from_template(
        """
        你是一个参数查询助手，根据用户输入的内容找出相关参数并且按照json格式返回。
        JSON字段如下：
        -"api_key":"YHPRGuiURJ5IzBX6IzqYJBUp8",
        -"name:"姓名",
        -"sex":"性别，0表示男，1表示女，如果用户告知了具体性别则填入用户告知的性别，如果没有告知具体性别则根据姓名判断性别",
        -"type:"日历类型，0表示农历，1表示公历",
        -"year":"出生年份，例如:1988",
        -"month":"出生月份，例如:1",
        -"day":"出生日，例如:1",
        -"hours":"出生时，例如:12",
        -"minute":"0",
        如果没有找到相关参数，则需要提醒用户告诉你这些内容，只返回json数据结构，不要有其他评论或者不相关的输出，用户输入如下：
        {query}
        """)
    parse = JsonOutputParser()
    prompt = prompt.partial(format_instructions=parse.get_format_instructions())
    chain = prompt | chatLLM | parse
    data = chain.invoke({"query": query})
    print("====八字查询参数====")
    print(data)
    result = requests.post(url, data)
    if result.status_code == 200:
        print("=====返回数据=====")
        print(result.json())
        try:
            json = result.json()
            returntring = f"八字为：{json['data']['bazi_info']['bazi']}"
            return returntring
        except Exception as e:
            return "八字查询失败，可能是你忘记咨询用户的姓名或者出生年月日时了。"
    else:
        return "技术错误，请告诉用户稍后再试"


@tool(description="只有当用户想占卜的时候才会使用这工具")
def zhanbu():
    url = f"https://api.yuanfenju.com/index.php/v1/Zhanbu/meiri"
    api_key = YUANFENJU_API_KEY
    result = requests.post(url, data={"api_key": api_key})
    if result.status_code == 200:
        returnstring = json.loads(result.text)  # 直接使用json()方法解析响应
        try:
            print("=====返回数据=====")
            print(returnstring)
            return returnstring
        except (KeyError, TypeError) as e:
            print(f"解析返回数据时出错：{e}")
            print(f"返回的数据结构：{returnstring}")
            return "获取卦象图片失败，请稍后再试"
    else:
        return "技术错误，请告诉用户稍后再试"


@tool(description="只有用户想要解梦时才会用到这个工具，需要用户输入梦境内容，如果没有输入梦境内容则不可用")
def jiemeng(query: str):
    url = f"https://api.yuanfenju.com/index.php/v1/Gongju/zhougong"
    api_key = YUANFENJU_API_KEY
    prompt = ChatPromptTemplate.from_template(
        """
        根据内容提取一个关键词，只返回关键词，内容为：{topic}
        """)
    prompt_vallue = prompt.invoke({"topic": query})
    keyword = chatLLM.invoke(prompt_vallue)
    print("提取的关键词为：",keyword.content)
    result = requests.post(url, data={
        "api_key": api_key,
        "title_zhougong": keyword.content
    })
    if result.status_code == 200:
        try:
            returnstring = json.loads(result.text)  # 直接使用json()方法解析响应
            print("=====返回数据=====")
            print(returnstring)
            return returnstring
        except (KeyError, TypeError) as e:
            print(f"解析返回数据时出错：{e}")
            return "解梦失败，请稍后再试"
    else:
        return "技术错误，请告诉用户稍后再试"

@tool(description="只有回答与今年运势或者蛇年运势相关的问题的时候，会使用这个工具且必须有用户的输入才能使用这个工具")
def get_info_from_local_db(query: str):
    client = Qdrant(
        QdrantClient(path="local_qdrant"),
        collection_name="local_docments",
        embeddings=embeddings, )
    retriever = client.as_retriever(search_type="mmr")
    result = retriever.get_relevant_documents(query)
    return result