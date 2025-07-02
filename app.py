import streamlit as st
import os
import json
import re
from datetime import datetime
from groq import Groq
from openai import OpenAI
from utils import execute_mcp_tool, handle_general_question
from postgres_mcp_handler import PostgresMCPHandler

class MCPClient:
    def __init__(self, db_config: dict):
        # Build PostgreSQL connection string
        conn_string = (
            f"postgresql://{db_config['username']}:{db_config['password']}"
            f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        )
        self.handler = PostgresMCPHandler(conn_string)

    def health_check(self):
        result = self.handler.health_check()
        return json.dumps(result), 0.01

    def list_tables(self, schema: str = "public"):
        result = self.handler.list_tables(schema)
        return json.dumps(result), 0.01

    def describe_table(self, table_name: str, schema: str = "public"):
        result = self.handler.describe_table(table_name, schema)
        return json.dumps(result), 0.01

    def find_relationships(self, table_name: str, schema: str = "public"):
        result = self.handler.find_relationships(table_name, schema)
        return json.dumps(result), 0.01

    def query(self, sql: str, parameters: list = None):
        result = self.handler.query(sql, parameters)
        return json.dumps(result), 0.01

    def stop_server(self):
        self.handler.close_pool()


st.set_page_config(
    page_title="MCP Postgres Database Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session State Initialization
st.session_state.setdefault("page", "connection")
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("connection_status", "disconnected")
st.session_state.setdefault("tables", [])
st.session_state.setdefault("schema_cache", {})
st.session_state.setdefault("mcp_client", None)
st.session_state.setdefault("groq_client", None)
st.session_state.setdefault("openai_client", None)
st.session_state.setdefault("ai_generated_questions", [])
st.session_state.setdefault("questions_generated", False)
st.session_state.setdefault("db_credentials", {
    "host": "localhost",
    "port": "5432",
    "database": "",
    "username": "",
    "password": ""
})
st.session_state.setdefault("groq_api_key", "")
st.session_state.setdefault("openai_api_key", "")
st.session_state.setdefault("selected_llm", "groq")
st.session_state.setdefault("latest_answer", None)


def create_mcp_client(credentials):
    try:
        client = MCPClient(credentials)
        return client
    except Exception as e:
        st.error(f"Failed to create MCP client: {e}")
        return None

def test_database_connection(credentials):
    try:
        client = create_mcp_client(credentials)
        if client:
            result, _ = client.health_check()
            data = json.loads(result)
            if isinstance(data, list) and data[0].get("healthy") == 1:
                return True, "Connection successful"
            else:
                return False, "Database health check failed"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"
    return False, "Unknown connection error"

def init_groq_client(api_key):
    try:
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="llama3-8b-8192",
            max_tokens=10
        )
        return client, "AI client initialized successfully"
    except Exception as e:
        return None, f"Failed to initialize AI client: {str(e)}"

def init_openai_client(api_key):
    try:
        client = OpenAI(api_key=api_key)
        client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello"}],
            model="gpt-3.5-turbo",
            max_tokens=10
        )
        return client, "OpenAI client initialized successfully"
    except Exception as e:
        return None, f"Failed to initialize OpenAI client: {str(e)}"

def analyze_database_schema():
    if not st.session_state.mcp_client:
        return False, "No database connection"
    try:
        result, _ = st.session_state.mcp_client.list_tables()
        tables_data = json.loads(result)
        table_names = [row.get("table_name") for row in tables_data if row.get("table_name")]
        st.session_state.tables = table_names
        schema_details = {}
        for table_name in table_names:
            desc_result, _ = st.session_state.mcp_client.describe_table(table_name)
            columns = json.loads(desc_result)
            schema_details[table_name] = columns
            st.session_state.schema_cache[table_name] = columns
            try:
                rel_result, _ = st.session_state.mcp_client.find_relationships(table_name)
                relationships = json.loads(rel_result)
                schema_details[f"{table_name}_relationships"] = relationships
            except:
                pass
        return True, schema_details
    except Exception as e:
        return False, f"Schema analysis failed: {str(e)}"

def generate_intelligent_questions(schema_details):
    selected_llm = st.session_state.selected_llm
    if (selected_llm == "groq" and not st.session_state.groq_client) or \
       (selected_llm == "openai" and not st.session_state.openai_client):
        return []
        
    schema_description = "Database Schema:\n\n"
    for table_name, columns in schema_details.items():
        if not table_name.endswith("_relationships") and isinstance(columns, list):
            schema_description += f"Table: {table_name}\nColumns:\n"
            for col in columns:
                schema_description += f"  - {col.get('column_name')} ({col.get('data_type')})\n"
    system_prompt = f"""
You are a database analyst. Based on this schema, generate 25 smart, diverse, natural-language questions providing insights.

{schema_description}

Return ONLY a JSON array: ["Q1", "Q2", ..., "Q25"]
"""
    try:
        if selected_llm == "groq":
            response = st.session_state.groq_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt.strip()}],
                model="llama3-8b-8192",
                temperature=0.3,
                max_tokens=2000
            )
            questions_text = response.choices[0].message.content.strip()
        else:  # OpenAI
            response = st.session_state.openai_client.chat.completions.create(
                messages=[{"role": "system", "content": system_prompt.strip()}],
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=2000
            )
            questions_text = response.choices[0].message.content.strip()
            
        start_idx = questions_text.find('[')
        end_idx = questions_text.rfind(']') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = questions_text[start_idx:end_idx]
            questions = json.loads(json_str)
            if isinstance(questions, list) and len(questions) > 0:
                return questions[:25]
        return []
    except Exception as e:
        st.error(f"Failed to generate questions: {e}")
        return []

with st.sidebar:
    st.title("Setup & Configuration")
    st.subheader("AI Configuration")
    
    selected_llm = st.radio("Select LLM Provider", ["groq", "openai"], index=0 if st.session_state.selected_llm == "groq" else 1)
    st.session_state.selected_llm = selected_llm
    
    if selected_llm == "groq":
        api_key_input = st.text_input("Groq API Key", value=st.session_state.groq_api_key, type="password")
        if api_key_input != st.session_state.groq_api_key:
            st.session_state.groq_api_key = api_key_input
            st.session_state.groq_client = None

        if st.session_state.groq_api_key and not st.session_state.groq_client:
            with st.spinner("Initializing Groq..."):
                client, message = init_groq_client(st.session_state.groq_api_key)
                if client:
                    st.session_state.groq_client = client
                    st.success(message)
                else:
                    st.error(message)
        elif st.session_state.groq_client:
            st.success("Groq Client Ready")
    else:  # OpenAI
        api_key_input = st.text_input("OpenAI API Key", value=st.session_state.openai_api_key, type="password")
        if api_key_input != st.session_state.openai_api_key:
            st.session_state.openai_api_key = api_key_input
            st.session_state.openai_client = None

        if st.session_state.openai_api_key and not st.session_state.openai_client:
            with st.spinner("Initializing OpenAI..."):
                client, message = init_openai_client(st.session_state.openai_api_key)
                if client:
                    st.session_state.openai_client = client
                    st.success(message)
                else:
                    st.error(message)
        elif st.session_state.openai_client:
            st.success("OpenAI Client Ready")

    st.divider()

    if st.session_state.page == "connection":
        st.subheader("Database Connection")
        with st.form("db_credentials_form", clear_on_submit=False):
            host = st.text_input("Host", value=st.session_state.db_credentials.get("host", "localhost"))
            port = st.text_input("Port", value=st.session_state.db_credentials.get("port", "5432"))
            database = st.text_input("Database Name", value=st.session_state.db_credentials.get("database", ""))
            username = st.text_input("Username", value=st.session_state.db_credentials.get("username", ""))
            password = st.text_input("Password", value=st.session_state.db_credentials.get("password", ""), type="password")
            submitted = st.form_submit_button("Connect to Database")
            if submitted:
                if (st.session_state.selected_llm == "groq" and not st.session_state.groq_client) or \
                   (st.session_state.selected_llm == "openai" and not st.session_state.openai_client):
                    st.error(f"Please initialize the {st.session_state.selected_llm.upper()} client first.")
                else:
                    st.session_state.db_credentials = {
                        "host": host, "port": port,
                        "database": database,
                        "username": username,
                        "password": password
                    }
                    if not all([database, username, password]):
                        st.error("Fill in all required fields")
                    else:
                        with st.spinner("Connecting..."):
                            success, message = test_database_connection(st.session_state.db_credentials)
                            if success:
                                st.session_state.mcp_client = create_mcp_client(st.session_state.db_credentials)
                                st.session_state.connection_status = "connected"
                                st.success(message)
                                st.session_state.page = "main"
                                st.rerun()
                            else:
                                st.error(message)
    else:
        st.subheader("Tools")
        if st.button("Load Tables"):
            if st.session_state.mcp_client:
                with st.spinner("Loading tables..."):
                    try:
                        result, _ = st.session_state.mcp_client.list_tables()
                        tables_data = json.loads(result)
                        table_names = [row.get("table_name") for row in tables_data if row.get("table_name")]
                        st.session_state.tables = table_names
                        st.success(f"Loaded {len(table_names)} tables.")
                    except Exception as e:
                        st.error(f"Failed to load tables: {e}")
        if st.button("Refresh Schema"):
            if st.session_state.mcp_client:
                with st.spinner("Refreshing schema..."):
                    st.session_state.schema_cache = {}
                    st.session_state.questions_generated = False
                    analyze_database_schema()
                    st.success("Schema refreshed")
                    st.rerun()
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()
        if st.button("Disconnect"):
            st.session_state.page = "connection"
            st.session_state.connection_status = "disconnected"
            st.session_state.mcp_client = None
            st.session_state.schema_cache = {}
            st.session_state.ai_generated_questions = []
            st.session_state.questions_generated = False
            st.session_state.chat_history = []
            st.rerun()

if st.session_state.page == "main":
    st.title("MCP Postgres Database Assistant")

    if not st.session_state.questions_generated:
        with st.spinner("Analyzing schema and generating questions..."):
            schema_success, schema_data = analyze_database_schema()
            if schema_success and ((st.session_state.selected_llm == "groq" and st.session_state.groq_client) or 
                                  (st.session_state.selected_llm == "openai" and st.session_state.openai_client)):
                questions = generate_intelligent_questions(schema_data)
                st.session_state.ai_generated_questions = questions
                st.session_state.questions_generated = True

    user_question = st.selectbox(
        "Pick an AI-generated question or type your own:",
        st.session_state.ai_generated_questions + ["Type your own"]
    )

    if user_question == "Type your own":
        user_question = st.text_input("Enter your question here")

    if st.button("Ask"):
        if user_question:
            with st.spinner("Thinking..."):
                answer = handle_general_question(user_question)

            st.session_state.chat_history.append({
                "type": "user",
                "content": user_question,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            st.session_state.chat_history.append({
                "type": "ai",
                "content": answer,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })

        st.session_state.latest_answer = answer
        st.rerun()

    if st.session_state.get("latest_answer"):
        st.markdown("#### Answer")
        with st.container():
            st.markdown(
                f"""
                <div style=\"background-color:#1c1c1c; padding:1rem; border-radius:0.5rem; border:1px solid #444;\">
                    {st.session_state.latest_answer}
                </div>
                """,
                unsafe_allow_html=True
            )

        if st.session_state.chat_history:
            with st.expander("Chat History", expanded=False):
                for msg in st.session_state.chat_history:
                    sender = "**User**" if msg["type"] == "user" else "** AI**"
                    st.markdown(f"{sender} ({msg['timestamp']}):\n\n{msg['content']}")

st.subheader("Advanced MCP Tools")
with st.expander("Run raw MCP tools"):
    tool = st.selectbox("Tool", ["list_tables", "describe_table", "find_relationships", "query", "health_check"])
    params = {}
    if tool in ["describe_table", "find_relationships"]:
        if st.session_state.tables:
            params["table_name"] = st.selectbox("Table Name", st.session_state.tables)
        else:
            params["table_name"] = st.text_input("Table Name")
    elif tool == "query":
        params["sql"] = st.text_area("SQL Query")

    if st.button("Run MCP Tool"):
        result, exec_time = execute_mcp_tool(tool, **params)
        st.write(f"Time: {exec_time:.2f}s")
        st.write(result)