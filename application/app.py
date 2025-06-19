import streamlit as st 
import chat
import utils
import json
import os
import mcp_config 
import asyncio
import logging
import sys

logging.basicConfig(
    level=logging.INFO,  # Default to INFO level
    format='%(filename)s:%(lineno)d | %(message)s',
    handlers=[
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger("streamlit")

os.environ["DEV"] = "true"  # Skip user confirmation of get_user_input

# title
st.set_page_config(page_title='Strands Agent', page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)

mode_descriptions = {
    "Agent": [
        "Uses Agent with Strands Agent SDK."
    ],
    "Agent (Chat)": [
        "A conversational Strands Agent."
    ]
}

with st.sidebar:
    st.title("🔮 Menu")
    
    st.markdown(
        "Implements various types of Agents using Strands Agent SDK. " 
        "For detailed code, please refer to [Github](https://github.com/kyopark2014/strands-agent)."
    )

    st.subheader("🐱 Conversation Type")
    
    # radio selection
    mode = st.radio(
        label="Select your desired conversation type: ",options=["Agent", "Agent (Chat)"], index=0
    )   
    st.info(mode_descriptions[mode][0])    
    # print('mode: ', mode)

    mcp_tools = chat.available_mcp_tools # available tool of mcp    
    
    mcp_selections = {}
    default_selections = ["basic", "filesystem", "use_aws", "aws document"]

    with st.expander("MCP Options Selection", expanded=True):            
        # Create two columns
        col1, col2 = st.columns(2)
        
        # Split options into two groups
        mid_point = len(mcp_tools) // 2
        first_half = mcp_tools[:mid_point]
        second_half = mcp_tools[mid_point:]
        
        # Display first group in the first column
        with col1:
            for option in first_half:
                default_value = option in default_selections
                mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)
        
        # Display second group in the second column
        with col2:
            for option in second_half:
                default_value = option in default_selections
                mcp_selections[option] = st.checkbox(option, key=f"mcp_{option}", value=default_value)

    selected_mcp_tools = [tool for tool in mcp_tools if mcp_selections.get(tool, False)]

    if mcp_selections["User Settings"]:
        mcp_info = st.text_area(
            "Enter MCP configuration in JSON format",
            value=mcp_config.mcp_user_config,
            height=150
        )
        logger.info(f"mcp_info: {mcp_info}")

        if mcp_info:
            mcp_config.mcp_user_config = json.loads(mcp_info)
            logger.info(f"mcp_user_config: {mcp_config.mcp_user_config}")
    
    # model selection box
    modelName = st.selectbox(
        '🖊️ Select Model to Use',
        ('Claude 4 Opus', 'Claude 4 Sonnet', 'Claude 3.7 Sonnet', 'Claude 3.5 Sonnet', 'Claude 3.0 Sonnet', 'Claude 3.5 Haiku'), index=3
    )

    # debug checkbox
    select_debugMode = st.checkbox('Debug Mode', value=True)
    debugMode = 'Enable' if select_debugMode else 'Disable'
    
    # extended thinking of claude 3.7 sonnet
    select_reasoning = st.checkbox('Reasoning', value=False)
    reasoningMode = 'Enable' if select_reasoning else 'Disable'
    logger.info(f"reasoningMode: {reasoningMode}")

    chat.update(modelName, reasoningMode, debugMode, selected_mcp_tools)
    
    st.success(f"Connected to {modelName}", icon="💚")
    clear_button = st.button("Clear Conversation", key="clear")
    # print('clear_button: ', clear_button)

st.title('🔮 '+ mode)  

if clear_button==True:
    chat.initiate()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.greetings = False

# Display chat messages from history on app rerun
def display_chat_messages():
    """Print message history
    @returns None
    """
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "images" in message:                
                for url in message["images"]:
                    logger.info(f"url: {url}")

                    file_name = url[url.rfind('/')+1:]
                    st.image(url, caption=file_name, use_container_width=True)            

display_chat_messages()

# Greet user
if not st.session_state.greetings:
    with st.chat_message("assistant"):
        intro = "Thank you for using Amazon Bedrock. You can enjoy comfortable conversations, and when you upload files, I can provide summaries."
        st.markdown(intro)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": intro})
        st.session_state.greetings = True

if clear_button or "messages" not in st.session_state:
    st.session_state.messages = []        
    
    st.session_state.greetings = False
    st.rerun()

    chat.clear_chat_history()
            
# Always show the chat input
if prompt := st.chat_input("Enter your message."):
    with st.chat_message("user"):  # display user message in chat message container
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})  # add user message to chat history
    prompt = prompt.replace('"', "").replace("'", "")
    logger.info(f"prompt: {prompt}")
    logger.info(f"is_updated: {chat.is_updated}")

    with st.chat_message("assistant"):
        if mode == 'Agent':
            history_mode = "Disable"
        elif mode == 'Agent (Chat)':
            history_mode = "Enable"

        tool_container = st.empty()
        status_container = st.empty()
        response_container = st.empty()          
        key_container = st.empty()
        
        response, image_urls = asyncio.run(chat.run_agent(prompt, history_mode, tool_container, status_container, response_container, key_container))
        
        for url in image_urls:
            logger.info(f"url: {url}")
            file_name = url[url.rfind('/')+1:]
            st.image(url, caption=file_name, use_container_width=True)      

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "images": image_urls if image_urls else []
        })
    