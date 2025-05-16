import autogen_agentchat.agents
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelInfo
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import  TextMentionTermination

mistral_vllm_model = OpenAIChatCompletionClient(
        model="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://g3.ai.qylis.com:8000/v1",
        api_key="NotRequired",
        model_info=ModelInfo(vision=False, function_calling=True, json_output=False, family="unknown", structured_output=True),
    )

agent = AssistantAgent(
        name="chat_agent",
        model_client=mistral_vllm_model, 
    )

agent_team = RoundRobinGroupChat([agent], termination_condition=TextMentionTermination("TERMINATE"))
config = agent_team.dump_component()
print(config.model_dump_json())
