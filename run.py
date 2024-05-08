import os
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.schema import AgentAction, AgentFinish
from typing import List


os.environ["OPENAI_API_KEY"] = "paste your api key here"
llm = ChatOpenAI(temperature=0)


question_agent_prompt_template = SystemMessagePromptTemplate.from_template(
    "You are a question generator agent. Your task is to create questions for the Answer Agent to answer."
)

#answer agent
answer_agent_prompt_template = SystemMessagePromptTemplate.from_template(
    "You are an answer agent. You answer questions posed by the Question Agent."
)


tools = [
    Tool(
        name="Answer Tool",
        func=lambda question: "This is a simulated answer to the question: " + question,
        description="A tool to get answers for questions."
    ),
]


question_agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    prompt_template=question_agent_prompt_template
)

answer_agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
    prompt_template=answer_agent_prompt_template
)


def multi_agent_interaction():
   
    question = "What is the capital of France?"
    question_response = question_agent({"input": question})

    print("Question Agent generated question:", question_response)

    
    answer_response = answer_agent({"input": question_response["output"]})

    print("Answer Agent's answer:", answer_response["output"])

multi_agent_interaction()
