import streamlit as st
import os
import asyncio
from datetime import date
from agents import Agent, Runner, WebSearchTool, FileSearchTool
from agents.mcp import MCPServerSse
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Ensure your OpenAI key is available from .env file
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
vector_store_id = os.environ.get("vector_store_id", "")
mcp_url_default = os.environ.get("MCP_URL", "")

# Initialize session state for chat history if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize search tool preferences if they don't exist
if "use_web_search" not in st.session_state:
    st.session_state.use_web_search = True
if "use_file_search" not in st.session_state:
    st.session_state.use_file_search = True
if "use_calendar" not in st.session_state:
    st.session_state.use_calendar = False
if "mcp_url" not in st.session_state:
    st.session_state.mcp_url = mcp_url_default

# Function to create agent with selected tools
async def create_unified_assistant():
    tools = []
    mcp_servers = []
    
    # Add web search tool
    if st.session_state.use_web_search:
        tools.append(WebSearchTool())
    
    # Add file search tool
    if st.session_state.use_file_search and vector_store_id:
        tools.append(FileSearchTool(
            max_num_results=3,
            vector_store_ids=[vector_store_id],
        ))
    
    # Add MCP calendar server
    if st.session_state.use_calendar and st.session_state.mcp_url:
        mcp_server = MCPServerSse(
            params={"url": st.session_state.mcp_url},
            cache_tools_list=True
        )
        await mcp_server.connect()
        mcp_servers.append(mcp_server)
    
    # Determine agent instructions based on enabled capabilities
    capabilities = []
    if st.session_state.use_web_search:
        capabilities.append("search the web")
    if st.session_state.use_file_search:
        capabilities.append("search through document vector stores")
    if st.session_state.use_calendar:
        capabilities.append("access and manage your Google Calendar")
    
    capabilities_text = ", ".join(capabilities) if capabilities else "assist you"
    today = date.today().strftime("%A, %B %d, %Y")
    
    instructions = f"""You are a helpful assistant that can {capabilities_text}.
    
    IMPORTANT: Today's date is {today}. Use this as the reference for "today", "this week", etc.
    
    Always cite your sources when responding to questions. Maintain the conversation context and refer to previous exchanges when appropriate.
    If you don't have enough information to answer a question, say so and suggest what additional information might help.
    
    Format your responses in a clear, readable manner using markdown formatting when appropriate.
    
    When working with calendar events:
    - Provide clear confirmations of any actions taken
    - Format dates and times clearly
    - Ask for clarification if event details are ambiguous
    """
    
    return Agent(
        name="Unified Assistant",
        instructions=instructions,
        tools=tools,
        mcp_servers=mcp_servers if mcp_servers else None,
    ), mcp_servers

# Async wrapper for running the agent with memory
async def get_assistant_response(question, history):
    # Create agent with current tool selections
    assistant, mcp_servers = await create_unified_assistant()
    
    try:
        # Combine history and current question to provide context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])  # Last 5 messages for context
        prompt = f"Context of our conversation:\n{context}\n\nCurrent question: {question}"
        
        result = await Runner.run(assistant, prompt)
        return result.final_output
    finally:
        # Clean up MCP connections
        for mcp_server in mcp_servers:
            await mcp_server.cleanup()

# Streamlit UI
st.set_page_config(page_title="Unified Assistant", layout="wide")
st.title("🔍 Unified Research & Calendar Assistant")
st.write("Ask me anything! I can search the web, access your documents, and manage your calendar.")

# Sidebar controls
st.sidebar.title("Assistant Settings")

# Tool selection toggles
st.sidebar.subheader("Select Capabilities")
web_search = st.sidebar.checkbox("Web Search", value=st.session_state.use_web_search, key="web_search_toggle")
file_search = st.sidebar.checkbox("Vector Store Search", value=st.session_state.use_file_search, key="file_search_toggle")
calendar = st.sidebar.checkbox("Google Calendar Access", value=st.session_state.use_calendar, key="calendar_toggle")

# Update session state when toggles change
if web_search != st.session_state.use_web_search:
    st.session_state.use_web_search = web_search
    
if file_search != st.session_state.use_file_search:
    st.session_state.use_file_search = file_search

if calendar != st.session_state.use_calendar:
    st.session_state.use_calendar = calendar

# MCP URL configuration (only show if calendar is enabled)
if st.session_state.use_calendar:
    st.sidebar.subheader("Calendar Configuration")
    mcp_url = st.sidebar.text_input(
        "MCP Server URL", 
        value=st.session_state.mcp_url,
        help="URL for your MCP calendar server"
    )
    if mcp_url != st.session_state.mcp_url:
        st.session_state.mcp_url = mcp_url

# Validate that at least one capability is selected
if not any([st.session_state.use_web_search, st.session_state.use_file_search, st.session_state.use_calendar]):
    st.sidebar.warning("⚠️ Please select at least one capability")

# Show which capabilities are active
st.sidebar.subheader("Active Capabilities")
active_caps = []
if st.session_state.use_web_search:
    active_caps.append("🌐 Web Search")
if st.session_state.use_file_search:
    active_caps.append("📚 Document Search")
if st.session_state.use_calendar:
    active_caps.append("📅 Calendar Access")

if active_caps:
    for cap in active_caps:
        st.sidebar.success(cap)
else:
    st.sidebar.error("No capabilities active")

# Conversation controls
st.sidebar.subheader("Conversation")
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()

# Display helpful examples
with st.sidebar.expander("Example Questions"):
    st.markdown("""
    **Research Questions:**
    - What are the latest developments in AI?
    - Find information about quantum computing
    - What's in my vector store documents?
    
    **Calendar Questions:**
    - What events do I have this week?
    - Schedule a meeting tomorrow at 2 PM
    - Show me my calendar for October 13, 2025
    - Cancel my 3 PM meeting today
    
    **Combined Questions:**
    - Research AI conferences and add the best one to my calendar
    - Find information about project deadlines in my documents
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("🐙 Made by the Lonely Octopus Team")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
user_question = st.chat_input("Ask your question")

if user_question:
    # Check if at least one capability is selected
    if not any([st.session_state.use_web_search, st.session_state.use_file_search, st.session_state.use_calendar]):
        st.error("⚠️ Please select at least one capability in the sidebar")
    else:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_question)
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                response_placeholder = st.empty()
                
                try:
                    # Get response from agent
                    response = asyncio.run(get_assistant_response(user_question, st.session_state.messages))
                    
                    # Update response placeholder
                    response_placeholder.markdown(response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_message = f"❌ Error: {str(e)}"
                    response_placeholder.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})