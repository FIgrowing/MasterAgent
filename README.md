## 项目介绍
该项目是一个AI Agent项目，主要针对的是算命、风水这一场景，利用大模型的推理能力，使用LangChain结合相关工具、RAG、Memory等进行Agent开发，并通过接入微软云TTS服务实现文本转语音、虚拟数字人等功能。该Agent主要功能就是进行八字测算、解梦、占卜、实时查询等功能。

## 项目地址
[https://github.com/FIgrowing/MasterAgent](https://github.com/FIgrowing/MasterAgent)

## 技术栈
1. Pyhton
2. LangChain
3. fastapi
4. Azure TTS&Avatar

## 项目架构图项目架构图
![画板](https://cdn.nlark.com/yuque/0/2025/jpeg/35810680/1749372578803-958c79cc-b31f-4b3b-9c9a-751c5948f507.jpeg)

## 模块分析
1. 多轮对话，使用RedisChatMessageHistory+ConversationBufferMemory，通过Redis+摘要压缩的方式持久化存储关键信息，实现上下文记忆功能解决了服务重启后对话记忆丢失的问题，同时节约了大量Token
2. RAG构建，使用document_loaders+RecursiveCharacterTextSplitter+chroma+DashScopeEmbeddings实现对网页、文本等内容进行在线解析、切分、向量化、存储功能，使得Agent具备实时学习功能
3. Prompt工程，运用角色定义、情绪判断、规范输出等方式优化提示词来提高回答准确性，同时使用Prompt Template特性实现动态插入变量，提高了提示词的动态性和可维护性
4. Tools工具开发，提供了谷歌搜索、RAG检索以及占卜、解梦等多样的工具丰富Agent能力

