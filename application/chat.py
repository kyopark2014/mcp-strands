import utils
import info
import traceback
import uuid
import logging
import sys

import contextlib
import mcp_config
import re

from contextlib import contextmanager
from typing import Dict, List, Optional
from botocore.config import Config
from strands import Agent
from strands.models import BedrockModel
from strands_tools import calculator, current_time, use_aws, python_repl
from strands.agent.conversation_manager import SlidingWindowConversationManager
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters
from urllib import parse
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("chat")

model_name = "Claude 3.7 Sonnet"
model_type = "claude"
debug_mode = "Enable"
model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
models = info.get_model_info(model_name)
bedrock_region = "us-west-2"
reasoning_mode = 'Disable'

available_mcp_tools = [
    "basic", "code interpreter", "aws document", "aws cost", "aws cli", 
    "use_aws", "aws cloudwatch", "aws storage", "tavily", "ArXiv", "wikipedia", 
    "filesystem", "terminal", "text editor", "context7", "puppeteer", 
    "playwright", "airbnb", 
    "pubmed", "chembl", "clinicaltrial", "arxiv-manual", "tavily-manual",                
    "aws_cloudwatch_logs", "User Settings"
]

mcp_selections = []
mcp_tools = []

# TAVILY_API_KEY check
config = utils.load_config()
tavily_key = config["tavily_api_key"] if "tavily_api_key" in config else None
if tavily_key is None:
    available_mcp_tools = [tool for tool in available_mcp_tools if tool not in ["tavily", "tavily-manual"]]
    logger.info(f"available_mcp_tools: {available_mcp_tools}")
    # Show popup when TAVILY_API_KEY is missing
    try:
        import streamlit as st
        st.error("Tavily Key is required!")
    except ImportError:
        # Print to console if streamlit is not available
        print("Tavily Key is required!")
        logger.warning("Tavily Key is required!")

is_updated = False
def update(modelName, reasoningMode, debugMode, selected_mcp_tools):    
    global model_name, model_id, model_type, reasoning_mode, debug_mode, mcp_tools, is_updated
    
    if model_name != modelName:
        model_name = modelName
        logger.info(f"model_name: {model_name}")
        
        model_id = models[0]["model_id"]
        model_type = models[0]["model_type"]

    if reasoningMode != reasoning_mode:
        reasoning_mode = reasoningMode
        logger.info(f"reasoning_mode: {reasoning_mode}")

    if debugMode != debug_mode:
        debug_mode = debugMode
        logger.info(f"debug_mode: {debug_mode}")
    
    if selected_mcp_tools != mcp_tools:
        mcp_tools = selected_mcp_tools
        is_updated = True
        init_mcp_clients()

    logger.info(f"mcp_tools: {mcp_tools}")

def traslation(chat, text, input_language, output_language):
    system = (
        "You are a helpful assistant that translates {input_language} to {output_language} in <article> tags." 
        "Put it in <result> tags."
    )
    human = "<article>{text}</article>"
    
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
    # print('prompt: ', prompt)
    
    chain = prompt | chat    
    try: 
        result = chain.invoke(
            {
                "input_language": input_language,
                "output_language": output_language,
                "text": text,
            }
        )
        
        msg = result.content
        # print('translated text: ', msg)
    except Exception:
        err_msg = traceback.format_exc()
        logger.info(f"error message: {err_msg}")     
        raise Exception ("Not able to request to LLM")

    return msg[msg.find('<result>')+8:len(msg)-9] # remove <result> tag

def initiate():
    global userId    
    userId = uuid.uuid4().hex
    logger.info(f"userId: {userId}")

def isKorean(text):
    # check korean
    pattern_hangul = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    word_kor = pattern_hangul.search(str(text))
    # print('word_kor: ', word_kor)

    if word_kor and word_kor != 'None':
        # logger.info(f"Korean: {word_kor}")
        return True
    else:
        # logger.info(f"Not Korean:: {word_kor}")
        return False
    
#########################################################
# Strands Agent 
#########################################################
def get_model():
    profile = models[0]
    if profile['model_type'] == 'nova':
        STOP_SEQUENCE = '"\n\n<thinking>", "\n<thinking>", " <thinking>"'
    elif profile['model_type'] == 'claude':
        STOP_SEQUENCE = "\n\nHuman:" 

    if model_type == 'claude':
        maxOutputTokens = 4096 # 4k
    else:
        maxOutputTokens = 5120 # 5k

    maxReasoningOutputTokens=64000
    thinking_budget = min(maxOutputTokens, maxReasoningOutputTokens-1000)

    if reasoning_mode=='Enable':
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=64000,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 1,
            additional_request_fields={
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": thinking_budget,
                }
            },
        )
    else:
        model = BedrockModel(
            boto_client_config=Config(
               read_timeout=900,
               connect_timeout=900,
               retries=dict(max_attempts=3, mode="adaptive"),
            ),
            model_id=model_id,
            max_tokens=maxOutputTokens,
            stop_sequences = [STOP_SEQUENCE],
            temperature = 0.1,
            top_p = 0.9,
            additional_request_fields={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
    return model

conversation_manager = SlidingWindowConversationManager(
    window_size=5,  # Reduced from 10 to 5 to decrease token usage
)

is_initiated = False
tools = []  

class MCPClientManager:
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        
    def add_client(self, name: str, command: str, args: List[str], env: dict[str, str] = {}) -> None:
        """Add a new MCP client"""
        self.clients[name] = MCPClient(lambda: stdio_client(
            StdioServerParameters(
                command=command, args=args, env=env
            )
        ))
    
    def remove_client(self, name: str) -> None:
        """Remove an MCP client"""
        if name in self.clients:
            del self.clients[name]
    
    @contextmanager
    def get_active_clients(self, active_clients: List[str]):
        """Manage active clients context"""
        logger.info(f"active_clients: {active_clients}")
        active_contexts = []
        try:
            for client_name in active_clients:
                logger.info(f"client_name: {client_name}")
                if client_name in self.clients:
                    active_contexts.append(self.clients[client_name])

            logger.info(f"active_contexts: {active_contexts}")
            if active_contexts:
                with contextlib.ExitStack() as stack:
                    for client in active_contexts:
                        stack.enter_context(client)
                    yield
            else:
                yield
        except Exception as e:
            logger.error(f"Error in MCP client context: {e}")
            raise

# Initialize MCP client manager
mcp_manager = MCPClientManager()

# Set up MCP clients
def init_mcp_clients():
    logger.info(f"available_mcp_tools: {available_mcp_tools}")
    
    for tool in available_mcp_tools:
        config = mcp_config.load_config_by_name(tool)
        # logger.info(f"config: {config}")

        # Skip if config is empty or doesn't have mcpServers
        if not config or "mcpServers" not in config:
            # logger.warning(f"No configuration found for tool: {tool}")
            continue

        # Get the first key from mcpServers
        server_key = next(iter(config["mcpServers"]))
        server_config = config["mcpServers"][server_key]
        
        name = tool  # Use tool name as client name
        command = server_config["command"]
        args = server_config["args"]
        env = server_config.get("env", {})  # Use empty dict if env is not present
        
        logger.info(f"name: {name}, command: {command}, args: {args}, env: {env}")        

        mcp_manager.add_client(name, command, args, env)

init_mcp_clients()

bedrock_region = config["region"] if "region" in config else "us-west-2"
projectName = config["projectName"] if "projectName" in config else "mcp-rag"

tool_list = []
def create_agent(history_mode, tool_container):
    global tools
    system = (
        "Your name is Seoyeon, and you are a thoughtful AI assistant who kindly answers questions."
        "You provide sufficient specific details appropriate to the situation." 
        "When you don't know the answer to a question, you honestly say you don't know."
    )

    model = get_model()

    try:
        # Convert tool names to actual tool objects
        tools = []

        # MCP tools - Limited to maximum 20 to control token usage
        max_mcp_tools = 20
        mcp_tools_to_process = mcp_tools[:max_mcp_tools] if len(mcp_tools) > max_mcp_tools else mcp_tools
        
        for mcp_tool in mcp_tools_to_process:
            logger.info(f"mcp_tool: {mcp_tool}")
            try:
                with mcp_manager.get_active_clients([mcp_tool]) as _:
                    logger.info(f"mcp_manager.clients: {mcp_manager.clients}")
                    
                    if mcp_tool in mcp_manager.clients:
                        client = mcp_manager.clients[mcp_tool]
                        mcp_tools_list = client.list_tools_sync()
                        logger.info(f"{mcp_tool}_tools: {len(mcp_tools_list)} tools loaded")
                        tools.extend(mcp_tools_list)
            except Exception as e:
                logger.error(f"Error loading MCP tool {mcp_tool}: {e}")
                continue

        logger.info(f"Total tools loaded: {len(tools)}")

        if history_mode == "Enable":
            logger.info("history_mode: Enable")
            agent = Agent(
                model=model,
                system_prompt=system,
                tools=tools,
                conversation_manager=conversation_manager
            )
        else:
            logger.info("history_mode: Disable")
            agent = Agent(
                model=model,
                system_prompt=system,
                tools=tools
                #max_parallel_tools=2
            )

        global tool_list
        tool_list = []
        for tool in tools:
            logger.info(f"tool: {tool}")
            # MCP tool
            if hasattr(tool, 'tool_name'):
                logger.info(f"MCP tool name: {tool.tool_name}")
                tool_list.append(tool.tool_name)
            
        logger.info(f"Tools: {tool_list}")

        if debug_mode == 'Enable':
            tool_container.info(f"Tools: {tool_list}")

    except Exception as e:
        logger.error(f"Error initializing MCP clients: {e}")
        # Use basic tools only when error occurs
        agent = Agent(
            model=model,
            system_prompt=system,
            tools=[calculator, current_time, use_aws, python_repl]
        )

    return agent

async def run_agent(question, history_mode, tool_container, status_container, response_container, key_container):
    final_response = ""
    current_response = ""
    image_urls = []

    global agent, is_initiated, is_updated
    if not is_initiated or is_updated:
        logger.info("create/update agent!")
        agent = create_agent(history_mode, tool_container)
        is_initiated = True
        is_updated = False
    else:
        if debug_mode == 'Enable':
            tool_container.info(f"Tools: {tool_list}")

    try:
        with mcp_manager.get_active_clients(mcp_tools) as _:
            agent_stream = agent.stream_async(question)
            
            tool_name = ""
            async for event in agent_stream:
                # logger.info(f"event: {event}")
                if "message" in event:
                    message = event["message"]
                    logger.info(f"message: {message}")
                    logger.info(f"content: {message["content"]}")

                    # role = message["role"]
                    # logger.info(f"role: {role}")

                    for content in message["content"]:                
                        if "text" in content:
                            logger.info(f"text: {content["text"]}")
                            current_response += '\n\n'
                            final_response = content["text"]


                        if "toolUse" in content:
                            tool_use = content["toolUse"]
                            logger.info(f"tool_use: {tool_use}")
                            
                            tool_name = tool_use["name"]
                            input = tool_use["input"]
                            
                            logger.info(f"tool_nmae: {tool_name}, arg: {input}")
                            if debug_mode == 'Enable':
                                status_container.info(f"tool name: {tool_name}, arg:: {input}")
                    
                        if "toolResult" in content:
                            tool_result = content["toolResult"]
                            logger.info(f"tool_result: {tool_result}")
                            if "content" in tool_result:
                                tool_content = tool_result["content"]
                                for content in tool_content:
                                    if "text" in content and debug_mode == 'Enable':
                                        response_container.info(f"tool result: {content["text"]}")

                if "data" in event:
                    text_data = event["data"]
                    current_response += text_data
                    key_container.markdown(current_response)
                    continue

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in run_agent: {error_msg}")
        
        if "Input is too long" in error_msg or "ValidationException" in error_msg:
            # Simple response when token limit is exceeded
            final_response = "Sorry, the input is too long to process. Please make your question simpler or reduce the number of tools you're using."
            if debug_mode == 'Enable':
                status_container.error("Input token limit exceeded - please reduce the number of tools or simplify your question.")
        else:
            final_response = f"An error occurred: {error_msg}"
            if debug_mode == 'Enable':
                status_container.error(f"Error: {error_msg}")

    return final_response, image_urls
            