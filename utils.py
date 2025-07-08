import json
import streamlit as st


def execute_mcp_tool(tool, **params):
    """Execute MCP tool with given parameters."""
    try:
        client = st.session_state.get("mcp_client")
        if not client:
            return "No MCP client available", 0.0

        import time
        start_time = time.time()

        if tool == "list_tables":
            result, _ = client.list_tables()
        elif tool == "describe_table":
            result, _ = client.describe_table(params.get("table_name", ""))
        elif tool == "find_relationships":
            result, _ = client.find_relationships(params.get("table_name", ""))
        elif tool == "query":
            result, _ = client.query(params.get("sql", ""))
        elif tool == "health_check":
            result, _ = client.health_check()
        else:
            result = f"Unknown tool: {tool}"

        exec_time = time.time() - start_time
        return result, exec_time

    except Exception as e:
        return f"Error executing tool: {str(e)}", 0.0


def handle_general_question(user_question):
    """MCP-compliant: LLM plans tool call → MCP runs it → LLM interprets result."""
    selected_llm = st.session_state.get("selected_llm", "groq")
    
    if selected_llm == "groq":
        llm_client = st.session_state.get("groq_client")
        model_name = "llama3-8b-8192"
    elif selected_llm == "openai":  # OpenAI
        llm_client = st.session_state.get("openai_client")
        model_name = "gpt-3.5-turbo"
    elif selected_llm == "Gemini":
        llm_client = st.session_state.get("gemini_client")
        model_name = "gemini-2.0-flash"
        
    mcp_client = st.session_state.get("mcp_client")
    schema_cache = st.session_state.get("schema_cache", {})

    if not llm_client or not mcp_client:
        return f"{selected_llm.upper()} client or MCP client is not initialized."

    # Step 1: Build schema context
    schema_summary = ""
    for table, columns in schema_cache.items():
        if isinstance(columns, list):
            schema_summary += f"Table: {table}\n"
            for col in columns:
                schema_summary += f"  - {col.get('column_name')} ({col.get('data_type')})\n"

    # Step 2: Ask AI to plan a tool call
    planning_prompt = f"""
You are an intelligent agent using the Model Context Protocol (MCP) to answer questions using tools.

You have access to the following tools:
- `list_tables`
- `describe_table` (requires `table_name`)
- `find_relationships` (requires `table_name`)
- `query` (requires `sql`)

Use ONLY the above tools. 
Based on this database schema and the user's question, output a JSON object like:

{{
  "tool": "query",
  "params": {{
    "sql": "SELECT ... "
  }}
}}

Only return the JSON. Do NOT explain or say anything else.

Schema:
{schema_summary}

User Question:
{user_question}
"""

    try:
        response = llm_client.chat.completions.create(
            messages=[{"role": "user", "content": planning_prompt}],
            model=model_name,
            temperature=0,
            max_tokens=1000,
        )
        tool_plan_raw = response.choices[0].message.content.strip()
        tool_plan = json.loads(tool_plan_raw)

        tool = tool_plan.get("tool")
        params = tool_plan.get("params", {})

        if not tool:
            return " AI could not determine a tool to use."

    except Exception as e:
        return f" Failed during tool planning: {e}"

    # Step 3: Run the tool
    try:
        if tool == "list_tables":
            result, _ = mcp_client.list_tables()
        elif tool == "describe_table":
            result, _ = mcp_client.describe_table(params.get("table_name", ""))
        elif tool == "find_relationships":
            result, _ = mcp_client.find_relationships(params.get("table_name", ""))
        elif tool == "query":
            result, _ = mcp_client.query(params.get("sql", ""))
        else:
            return f" Unsupported tool: {tool}"
    except Exception as e:
        return f" Tool execution failed: {e}"

    # Step 4: Send result back to AI to interpret
    interpretation_prompt = f"""
You are a smart AI that just used the MCP tool `{tool}`.

Now, interpret the result below and explain it to the user in simple, helpful language. 
"Use PostgreSQL-compatible SQL syntax. Avoid MySQL-only functions."
Don't show SQL or raw JSON. Just give the answer in plain English.
"Make sure questions reflect actual foreign-key relationships and available columns. Avoid assuming that one table contains information that must be joined from another."


User Question:
{user_question}

Result:
{result}
"""

    try:
        response = llm_client.chat.completions.create(
            messages=[{"role": "user", "content": interpretation_prompt}],
            model=model_name,
            temperature=0.7,
            max_tokens=1000,
        )
        final_answer = response.choices[0].message.content.strip()
        return final_answer
    except Exception as e:
        return f" Failed to interpret tool result: {e}"
    