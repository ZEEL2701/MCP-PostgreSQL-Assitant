import json
import streamlit as st
import re
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("mcp-postgres-assistant")


def execute_mcp_tool(tool, **params):
    """Execute MCP tool with given parameters."""
    try:
        client = st.session_state.get("mcp_client")
        if not client:
            logger.error("No MCP client available")
            return "No MCP client available", 0.0

        import time
        start_time = time.time()
        
        logger.info(f"Executing MCP tool: {tool} with params: {params}")

        if tool == "list_tables":
            result, _ = client.list_tables()
            logger.info(f"list_tables result: {result[:200]}...")
        elif tool == "describe_table":
            table_name = params.get("table_name", "")
            result, _ = client.describe_table(table_name)
            logger.info(f"describe_table for {table_name} result: {result[:200]}...")
        elif tool == "find_relationships":
            table_name = params.get("table_name", "")
            result, _ = client.find_relationships(table_name)
            logger.info(f"find_relationships for {table_name} result: {result[:200]}...")
        elif tool == "query":
            sql = params.get("sql", "")
            logger.info(f"Executing SQL query: {sql}")
            result, _ = client.query(sql)
            logger.info(f"Query result: {result[:200]}...")
        elif tool == "health_check":
            result, _ = client.health_check()
            logger.info(f"health_check result: {result}")
        else:
            result = f"Unknown tool: {tool}"
            logger.error(f"Unknown tool requested: {tool}")

        exec_time = time.time() - start_time
        logger.info(f"Tool {tool} executed in {exec_time:.2f} seconds")
        return result, exec_time

    except Exception as e:
        error_msg = f"Error executing tool {tool}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return error_msg, 0.0


def extract_table_names_from_question(question, available_tables):
    """
    Extract potential table names from the user's question by matching against available tables.
    """
    mentioned_tables = []

    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()

    for table in available_tables:
        # Check for exact table name match (case insensitive)
        if table.lower() in question_lower:
            mentioned_tables.append(table)
        # Check for plural forms (simple pluralization by adding 's')
        elif table.lower() + 's' in question_lower:
            mentioned_tables.append(table)

    return mentioned_tables


def validate_tables_and_get_relationships(tables, mcp_client):
    """
    Validate if tables exist and get relationships between them.

    Args:
        tables: List of table names to validate
        mcp_client: The PostgreSQL MCP client

    Returns:
        tuple: (existing_tables, missing_tables, relationships, table_schemas)
    """
    existing_tables = []
    missing_tables = []
    relationships = []
    table_schemas = {}

    # Get all available tables first
    result, _ = mcp_client.list_tables()
    all_tables_data = json.loads(result)
    available_tables = [row.get("table_name")
                        for row in all_tables_data if row.get("table_name")]

    # Check which tables exist
    for table in tables:
        if table in available_tables:
            existing_tables.append(table)
        else:
            missing_tables.append(table)

    # Get schema for existing tables
    for table in existing_tables:
        result, _ = mcp_client.describe_table(table)
        table_schemas[table] = json.loads(result)

    # Get relationships between tables
    for i in range(len(existing_tables)):
        for j in range(i+1, len(existing_tables)):
            table1 = existing_tables[i]
            table2 = existing_tables[j]

            # Check relationships from table1 to table2
            result, _ = mcp_client.find_relationships(table1)
            relations = json.loads(result)
            for relation in relations:
                if relation.get("foreign_table") == table2:
                    relationships.append({
                        "source_table": table1,
                        "source_column": relation.get("column_name"),
                        "target_table": table2,
                        "target_column": relation.get("foreign_column")
                    })

            # Check relationships from table2 to table1
            result, _ = mcp_client.find_relationships(table2)
            relations = json.loads(result)
            for relation in relations:
                if relation.get("foreign_table") == table1:
                    relationships.append({
                        "source_table": table2,
                        "source_column": relation.get("column_name"),
                        "target_table": table1,
                        "target_column": relation.get("foreign_column")
                    })

    return existing_tables, missing_tables, relationships, table_schemas


def get_all_table_schemas(mcp_client):
    """
    Get schemas for all tables in the database.

    Args:
        mcp_client: The PostgreSQL MCP client

    Returns:
        dict: Dictionary mapping table names to their schemas
    """
    table_schemas = {}

    # Get all available tables
    result, _ = mcp_client.list_tables()
    all_tables_data = json.loads(result)
    available_tables = [row.get("table_name")
                        for row in all_tables_data if row.get("table_name")]

    # Get schema for each table
    for table in available_tables:
        result, _ = mcp_client.describe_table(table)
        table_schemas[table] = json.loads(result)

    return table_schemas


def find_datetime_columns(table_schemas):
    """
    Find all date/time columns in the database.

    Args:
        table_schemas: Dictionary mapping table names to their schemas

    Returns:
        dict: Dictionary mapping table names to lists of date/time column names
    """
    datetime_columns = {}

    for table_name, columns in table_schemas.items():
        datetime_cols = []
        for column in columns:
            data_type = column.get("data_type", "").lower()
            if any(dt_type in data_type for dt_type in ["date", "time", "timestamp"]):
                datetime_cols.append(column.get("column_name"))

        if datetime_cols:
            datetime_columns[table_name] = datetime_cols

    return datetime_columns


def extract_tables_from_sql(sql):
    """
    Extract table names from an SQL query.

    Args:
        sql: SQL query string

    Returns:
        list: List of table names referenced in the query
    """
    # Extract table names from FROM clause
    from_pattern = re.compile(
        r'FROM\s+([a-zA-Z0-9_,\s]+)(?:WHERE|GROUP BY|ORDER BY|LIMIT|$)', re.IGNORECASE | re.DOTALL)
    from_match = from_pattern.search(sql)
    tables = []

    if from_match:
        from_clause = from_match.group(1).strip()
        # Split by commas for multiple tables
        table_parts = from_clause.split(',')
        for part in table_parts:
            # Extract table name (handle aliases)
            table_name = part.strip().split(' ')[0].strip()
            if table_name:
                tables.append(table_name)

    # Extract table names from JOIN clauses
    join_pattern = re.compile(r'JOIN\s+([a-zA-Z0-9_]+)', re.IGNORECASE)
    join_matches = join_pattern.findall(sql)
    tables.extend(join_matches)

    return tables


def verify_column_exists(table_schemas, table_name, column_name):
    """
    Verify if a column exists in a table.

    Args:
        table_schemas: Dictionary mapping table names to their schemas
        table_name: Name of the table
        column_name: Name of the column to check

    Returns:
        tuple: (exists, similar_column)
        - exists: Boolean indicating if the column exists
        - similar_column: Name of a similar column if exists, None otherwise
    """
    if table_name not in table_schemas:
        return False, None

    # Check for exact match
    for column in table_schemas[table_name]:
        if column.get("column_name").lower() == column_name.lower():
            return True, column.get("column_name")

    # Check for similar columns (simple similarity check)
    for column in table_schemas[table_name]:
        col_name = column.get("column_name").lower()
        if (column_name.lower() in col_name) or (col_name in column_name.lower()):
            return False, column.get("column_name")

    return False, None


def extract_columns_from_sql(sql):
    """
    Extract table and column names from an SQL query.

    Args:
        sql: SQL query string

    Returns:
        list: List of tuples (table_name, column_name)
    """
    # This is a simplified extraction and might not catch all cases
    # Extract table.column patterns
    table_column_pattern = re.compile(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)')
    table_columns = table_column_pattern.findall(sql)

    # Extract column names from SELECT clause
    select_pattern = re.compile(
        r'SELECT\s+(.+?)\s+FROM', re.IGNORECASE | re.DOTALL)
    select_match = select_pattern.search(sql)
    if select_match:
        select_columns = select_match.group(1).split(',')
        # Extract column names without table prefix
        for col in select_columns:
            col = col.strip()
            # Skip if it's an expression or has a table prefix
            if '(' not in col and '.' not in col and col != '*':
                # Add to the list with None as table name
                table_columns.append((None, col))

    # Extract columns from EXTRACT functions (for date/time operations)
    extract_pattern = re.compile(
        r'EXTRACT\s*\(\s*\w+\s+FROM\s+([a-zA-Z0-9_\.]+)\s*\)', re.IGNORECASE)
    extract_matches = extract_pattern.findall(sql)
    for col in extract_matches:
        if '.' in col:
            table_name, column_name = col.split('.', 1)
            table_columns.append((table_name, column_name))
        else:
            table_columns.append((None, col))

    # Extract columns from WHERE clause
    where_pattern = re.compile(
        r'WHERE\s+(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', re.IGNORECASE | re.DOTALL)
    where_match = where_pattern.search(sql)
    if where_match:
        where_clause = where_match.group(1)
        # Extract column names from WHERE clause
        where_columns = re.findall(
            r'([a-zA-Z0-9_]+)(?:\s*=|\s*<|\s*>|\s*LIKE|\s+IN|\s+BETWEEN)', where_clause, re.IGNORECASE)
        for col in where_columns:
            if '.' not in col and col not in ['AND', 'OR', 'NOT']:
                table_columns.append((None, col))

    return table_columns


def validate_sql_query(sql, table_schemas):
    """
    Validate an SQL query against table schemas.

    Args:
        sql: SQL query string
        table_schemas: Dictionary mapping table names to their schemas

    Returns:
        tuple: (is_valid, error_message, corrected_sql)
    """
    logger.info(f"Validating SQL query: {sql}")
    
    # Extract table and column references from the SQL
    table_columns = extract_columns_from_sql(sql)
    logger.info(f"Extracted table columns: {table_columns}")

    # Extract tables mentioned in the FROM clause
    from_tables = extract_tables_from_sql(sql)
    logger.info(f"Tables in FROM clause: {from_tables}")

    corrected_sql = sql
    errors = []

    # Find all date/time columns in the database
    datetime_columns = find_datetime_columns(table_schemas)
    logger.info(f"Date/time columns: {datetime_columns}")

    # Check for tables referenced in column expressions but not included in the FROM clause
    referenced_tables = set()
    for table_name, _ in table_columns:
        if table_name:
            referenced_tables.add(table_name)

    missing_from_tables = referenced_tables - set(from_tables)
    if missing_from_tables:
        logger.warning(f"Tables referenced but not in FROM clause: {missing_from_tables}")
        for missing_table in missing_from_tables:
            if missing_table in table_schemas:
                # Table exists but is not in the FROM clause
                # Try to fix by adding it to the FROM clause
                if "FROM" in sql.upper():
                    # Add the missing table to the FROM clause
                    from_pattern = re.compile(
                        r'(FROM\s+[a-zA-Z0-9_,\s]+)', re.IGNORECASE)
                    from_match = from_pattern.search(sql)
                    if from_match:
                        from_clause = from_match.group(1)
                        corrected_sql = corrected_sql.replace(
                            from_clause, f"{from_clause}, {missing_table}")
                        error_msg = f"Table '{missing_table}' was referenced but not included in the FROM clause. Added to FROM clause."
                        errors.append(error_msg)
                        logger.info(f"Fixed SQL: {error_msg}")
                        logger.info(f"Corrected SQL: {corrected_sql}")

    # Check for date/time operations
    if "EXTRACT" in sql.upper() and "QUARTER" in sql.upper():
        logger.info("Detected quarterly analysis query with EXTRACT(QUARTER FROM ...)")
        # This is a quarterly analysis query, check if we're using a valid date column
        extract_pattern = re.compile(
            r'EXTRACT\s*\(\s*QUARTER\s+FROM\s+([a-zA-Z0-9_\.]+)\s*\)', re.IGNORECASE)
        extract_matches = extract_pattern.findall(sql)
        logger.info(f"Found EXTRACT QUARTER expressions: {extract_matches}")

        for col in extract_matches:
            table_name = None
            column_name = col

            if '.' in col:
                parts = col.split('.', 1)
                table_name = parts[0]
                column_name = parts[1]
                logger.info(f"Parsed column reference: table={table_name}, column={column_name}")
            else:
                logger.info(f"No table specified for column: {column_name}")

            # If table name is specified, check if the column exists in that table
            if table_name:
                exists, similar_column = verify_column_exists(
                    table_schemas, table_name, column_name)
                if not exists:
                    logger.warning(f"Column {column_name} does not exist in table {table_name}")
                    # Check if there's a similar date/time column in this table
                    if table_name in datetime_columns and datetime_columns[table_name]:
                        suggested_col = datetime_columns[table_name][0]
                        errors.append(
                            f"Column '{column_name}' does not exist in table '{table_name}'. Did you mean '{suggested_col}'?")
                        corrected_sql = corrected_sql.replace(
                            f"{table_name}.{column_name}", f"{table_name}.{suggested_col}")
                    else:
                        errors.append(
                            f"Column '{column_name}' does not exist in table '{table_name}' and no date/time columns were found.")
            else:
                # If no table name is specified, check all tables for a matching date/time column
                found = False
                for table, dt_cols in datetime_columns.items():
                    if column_name in dt_cols:
                        found = True
                        break

                if not found:
                    # Suggest the first date/time column from any table
                    for table, dt_cols in datetime_columns.items():
                        if dt_cols:
                            suggested_table = table
                            suggested_col = dt_cols[0]
                            errors.append(
                                f"Column '{column_name}' was not found. Did you mean '{suggested_table}.{suggested_col}'?")
                            corrected_sql = corrected_sql.replace(
                                column_name, f"{suggested_table}.{suggested_col}")
                            break

    # Validate each table.column reference
    for table_name, column_name in table_columns:
        if table_name is None:
            # Skip validation for columns without table prefix
            continue

        if table_name not in table_schemas:
            errors.append(f"Table '{table_name}' does not exist")
            continue

        exists, similar_column = verify_column_exists(
            table_schemas, table_name, column_name)
        if not exists:
            if similar_column:
                errors.append(
                    f"Column '{column_name}' does not exist in table '{table_name}'. Did you mean '{similar_column}'?")
                # Replace the incorrect column with the similar one
                corrected_sql = corrected_sql.replace(
                    f"{table_name}.{column_name}", f"{table_name}.{similar_column}")
            else:
                errors.append(
                    f"Column '{column_name}' does not exist in table '{table_name}'")

    is_valid = len(errors) == 0
    error_message = "\n".join(errors) if errors else ""

    return is_valid, error_message, corrected_sql


def handle_general_question(user_question):
    """MCP-compliant: LLM plans tool call → MCP runs it → LLM interprets result."""
    selected_llm = st.session_state.get("selected_llm", "groq")

    if selected_llm == "groq":
        llm_client = st.session_state.get("groq_client")
        model_name = "llama3-8b-8192"
    elif selected_llm == "openai":  # OpenAI
        llm_client = st.session_state.get("openai_client")
        model_name = "gpt-4o"
    else:  # Gemini
        llm_client = st.session_state.get("gemini_client")
        model_name = "gemini-2.0-flash"  # Updated to use the latest model

    mcp_client = st.session_state.get("mcp_client")
    schema_cache = st.session_state.get("schema_cache", {})

    if not llm_client or not mcp_client:
        return f"{selected_llm.upper()} client or MCP client is not initialized."

    # Get all available tables
    result, _ = mcp_client.list_tables()
    all_tables_data = json.loads(result)
    available_tables = [row.get("table_name")
                        for row in all_tables_data if row.get("table_name")]

    # Extract potential table names from the question
    mentioned_tables = extract_table_names_from_question(
        user_question, available_tables)

    # Get all table schemas for validation
    all_table_schemas = get_all_table_schemas(mcp_client)

    # Find all date/time columns for special handling
    datetime_columns = find_datetime_columns(all_table_schemas)

    # If we have potential tables mentioned, validate them and get relationships
    enhanced_schema_context = ""
    if mentioned_tables:
        existing_tables, missing_tables, relationships, table_schemas = validate_tables_and_get_relationships(
            mentioned_tables, mcp_client
        )

        # If there are missing tables, inform the user
        if missing_tables:
            return f"The following tables mentioned in your question do not exist in the database: {', '.join(missing_tables)}"

        # Build enhanced schema context with relationships
        enhanced_schema_context = "Tables and their schemas:\n\n"
        for table, schema in table_schemas.items():
            enhanced_schema_context += f"Table: {table}\n"
            for col in schema:
                enhanced_schema_context += f"  - {col.get('column_name')} ({col.get('data_type')})\n"

        if relationships:
            enhanced_schema_context += "\nRelationships between tables:\n"
            for rel in relationships:
                enhanced_schema_context += f"  - {rel['source_table']}.{rel['source_column']} → {rel['target_table']}.{rel['target_column']}\n"

    # Add date/time columns information
    datetime_context = "\nDate/Time columns (for time-based analysis):\n"
    for table, columns in datetime_columns.items():
        if columns:
            datetime_context += f"  - {table}: {', '.join(columns)}\n"

    # Step 1: Build schema context (use enhanced context if available, otherwise use all tables)
    schema_summary = enhanced_schema_context if enhanced_schema_context else ""
    if not schema_summary:
        for table, columns in schema_cache.items():
            if isinstance(columns, list):
                schema_summary += f"Table: {table}\n"
                for col in columns:
                    schema_summary += f"  - {col.get('column_name')} ({col.get('data_type')})\n"

    # Add date/time columns to the schema summary
    schema_summary += datetime_context

    # Step 2: Ask AI to plan a tool call with improved prompt
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

IMPORTANT: Return ONLY the raw JSON object without any markdown formatting, code blocks, or explanations.

When generating SQL queries:
1. Use PostgreSQL-compatible syntax
2. For multi-table queries, use proper JOIN conditions based on the relationships
3. Use aliases for table names to make the query more readable
4. Include appropriate WHERE clauses to filter data as needed
5. Use ORDER BY for sorting when appropriate
6. Use LIMIT to restrict large result sets
7. IMPORTANT: Make sure all column names exist in the tables you're querying
8. For date/time queries, check if the column is actually a timestamp/date type
9. For quarterly analysis, use EXTRACT(QUARTER FROM date_column) syntax with a valid date column
10. Always verify column names match exactly what's in the schema
11. For time-based analysis, use one of the date/time columns listed in the schema
12. CRITICAL: Always include all referenced tables in the FROM clause or JOIN statements

Only return the JSON. Do NOT explain or say anything else.

Schema:
{schema_summary}

User Question:
{user_question}
"""

    try:
        if selected_llm in ["groq", "openai"]:
            logger.info(
                f"Sending prompt to {selected_llm} model {model_name}:\n{planning_prompt[:500]}...")

            response = llm_client.chat.completions.create(
                messages=[{"role": "user", "content": planning_prompt}],
                model=model_name,
                temperature=0,
                max_tokens=2000,  # Increased max_tokens to avoid truncation
            )
            tool_plan_raw = response.choices[0].message.content.strip()

            logger.info(
                f"Raw response from {selected_llm} model {model_name}:\n{tool_plan_raw}")

            # Extract JSON from OpenAI's response which might contain markdown or other text
            # This is especially important for GPT-4o which often returns markdown code blocks
            json_match = re.search(
                r'```(?:json)?\s*({.*?})\s*```', tool_plan_raw, re.DOTALL)
            if json_match:
                tool_plan_raw = json_match.group(1)
                logger.info(
                    f"Extracted JSON from markdown code block: {tool_plan_raw[:200]}...")
            else:
                # Try to find JSON object without code blocks
                json_match = re.search(r'({[\s\S]*?})', tool_plan_raw)
                if json_match:
                    tool_plan_raw = json_match.group(1)
                    logger.info(
                        f"Extracted JSON from text: {tool_plan_raw[:200]}...")

            # Clean up any potential markdown or text artifacts
            tool_plan_raw = tool_plan_raw.strip()
        else:  # Gemini
            logger.info(
                f"Sending prompt to {selected_llm} model {model_name}:\n{planning_prompt[:500]}...")

            response = llm_client.generate_content(planning_prompt)
            tool_plan_raw = response.text.strip()

            logger.info(
                f"Raw response from {selected_llm} model {model_name}:\n{tool_plan_raw}")

            # Extract JSON from Gemini's response which might contain markdown or other text
            json_match = re.search(
                r'```(?:json)?\s*({.*?})\s*```', tool_plan_raw, re.DOTALL)
            if json_match:
                tool_plan_raw = json_match.group(1)
                logger.info(
                    f"Extracted JSON from markdown code block: {tool_plan_raw[:200]}...")
            else:
                # Try to find JSON object without code blocks
                json_match = re.search(r'({[\s\S]*?})', tool_plan_raw)
                if json_match:
                    tool_plan_raw = json_match.group(1)
                    logger.info(
                        f"Extracted JSON from text: {tool_plan_raw[:200]}...")

            # Clean up any potential markdown or text artifacts
            tool_plan_raw = tool_plan_raw.strip()

        try:
            # Try to fix common JSON issues before parsing
            # 1. Fix truncated JSON by adding missing closing braces
            if tool_plan_raw.count('{') > tool_plan_raw.count('}'):
                missing_braces = tool_plan_raw.count(
                    '{') - tool_plan_raw.count('}')
                tool_plan_raw += '}' * missing_braces
                logger.info(
                    f"Fixed truncated JSON by adding {missing_braces} closing braces")

            # 2. Fix missing quotes around property names
            tool_plan_raw = re.sub(
                r'([{,])\s*(\w+):', r'\1"\2":', tool_plan_raw)

            # 3. Fix trailing commas
            tool_plan_raw = re.sub(r',\s*}', '}', tool_plan_raw)

            logger.info(f"Final JSON to parse: {tool_plan_raw}")
            tool_plan = json.loads(tool_plan_raw)
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {e}. Raw response: ```{tool_plan_raw[:300]}...```"
            logger.error(
                f"JSON parsing error: {e}\nFull raw response: {tool_plan_raw}")

            # Try a more aggressive approach to extract valid JSON
            try:
                # Look for anything that looks like a complete JSON object
                all_json_objects = re.findall(r'({[^{]*?})', tool_plan_raw)
                if all_json_objects:
                    for json_obj in all_json_objects:
                        try:
                            tool_plan = json.loads(json_obj)
                            if "tool" in tool_plan and "params" in tool_plan:
                                logger.info(
                                    f"Successfully extracted valid JSON object: {json_obj}")
                                break
                        except:
                            continue
                    else:
                        return f"{error_msg}\n\nCould not extract a valid JSON object. Please try again with a simpler query or a different LLM provider."
                else:
                    return f"{error_msg}\n\nThis is likely due to the LLM not returning a valid JSON object. Please try again or try with a different LLM provider."
            except Exception as ex:
                return f"{error_msg}\n\nFailed to recover: {str(ex)}"

        tool = tool_plan.get("tool")
        params = tool_plan.get("params", {})

        if not tool:
            return " AI could not determine a tool to use."

    except Exception as e:
        return f" Failed during tool planning: {e}"

    # Step 3: Run the tool with validation for SQL queries
    try:
        if tool == "list_tables":
            result, _ = mcp_client.list_tables()
        elif tool == "describe_table":
            table_name = params.get("table_name", "")
            if table_name not in available_tables:
                return f"Table '{table_name}' does not exist in the database."
            result, _ = mcp_client.describe_table(table_name)
        elif tool == "find_relationships":
            table_name = params.get("table_name", "")
            if table_name not in available_tables:
                return f"Table '{table_name}' does not exist in the database."
            result, _ = mcp_client.find_relationships(table_name)
        elif tool == "query":
            sql = params.get("sql", "")

            # Special handling for quarterly analysis queries
            if "EXTRACT" in sql.upper() and "QUARTER" in sql.upper():
                # Check if we're using a valid date column
                extract_pattern = re.compile(
                    r'EXTRACT\s*\(\s*QUARTER\s+FROM\s+([a-zA-Z0-9_\.]+)\s*\)', re.IGNORECASE)
                extract_matches = extract_pattern.findall(sql)

                for col in extract_matches:
                    table_name = None
                    column_name = col

                    if '.' in col:
                        parts = col.split('.', 1)
                        table_name = parts[0]
                        column_name = parts[1]

                    # If no table is specified or column doesn't exist, try to find a suitable date column
                    if not table_name or not any(verify_column_exists(all_table_schemas, t, column_name)[0] for t in all_table_schemas):
                        # Find a suitable date column from any table
                        for t, dt_cols in datetime_columns.items():
                            if dt_cols:
                                # Replace the column with a valid date column
                                sql = sql.replace(
                                    f"EXTRACT(QUARTER FROM {col})", f"EXTRACT(QUARTER FROM {t}.{dt_cols[0]})")
                                break

            # Check for tables referenced in column expressions but not included in the FROM clause
            referenced_tables = set()
            table_columns = extract_columns_from_sql(sql)
            for table_name, _ in table_columns:
                if table_name:
                    referenced_tables.add(table_name)

            from_tables = extract_tables_from_sql(sql)
            missing_from_tables = referenced_tables - set(from_tables)
            
            logger.info(f"Referenced tables: {referenced_tables}")
            logger.info(f"Tables in FROM clause: {from_tables}")
            logger.info(f"Missing tables: {missing_from_tables}")

            if missing_from_tables:
                # There are tables referenced but not in the FROM clause
                # Try to fix by adding them to the FROM clause
                logger.warning(f"Tables referenced but not in FROM clause: {missing_from_tables}")
                for missing_table in missing_from_tables:
                    if missing_table in all_table_schemas:
                        if "FROM" in sql.upper():
                            # Add the missing table to the FROM clause
                            from_pattern = re.compile(
                                r'(FROM\s+[a-zA-Z0-9_,\s]+)', re.IGNORECASE)
                            from_match = from_pattern.search(sql)
                            if from_match:
                                from_clause = from_match.group(1)
                                sql = sql.replace(
                                    from_clause, f"{from_clause}, {missing_table}")
                                logger.info(f"Added missing table {missing_table} to FROM clause")
                                logger.info(f"Updated SQL: {sql}")

            # Validate the SQL query against the schema
            is_valid, error_message, corrected_sql = validate_sql_query(
                sql, all_table_schemas)

            if not is_valid:
                # If validation failed, try to fix the query with the LLM
                logger.warning(f"SQL validation failed: {error_message}")
                fix_prompt = f"""
You are a PostgreSQL expert. The following SQL query has errors:

```sql
{sql}
```

Error details:
{error_message}

Database schema:
{schema_summary}

Please fix the SQL query to work with this schema. Only return the corrected SQL query, nothing else.
"""
                logger.info(f"Sending SQL fix prompt to {selected_llm} model")
                
                if selected_llm in ["groq", "openai"]:
                    fix_response = llm_client.chat.completions.create(
                        messages=[{"role": "user", "content": fix_prompt}],
                        model=model_name,
                        temperature=0,
                        max_tokens=1000,
                    )
                    fixed_sql = fix_response.choices[0].message.content.strip()
                    logger.info(f"Raw SQL fix response from {selected_llm}: {fixed_sql[:200]}...")
                else:  # Gemini
                    fix_response = llm_client.generate_content(fix_prompt)
                    fixed_sql = fix_response.text.strip()
                    logger.info(f"Raw SQL fix response from {selected_llm}: {fixed_sql[:200]}...")

                # Extract SQL from markdown code blocks if present
                sql_match = re.search(
                    r'```(?:sql)?\s*(.*?)\s*```', fixed_sql, re.DOTALL)
                if sql_match:
                    fixed_sql = sql_match.group(1).strip()
                    logger.info(f"Extracted SQL from code block: {fixed_sql}")
                
                logger.info(f"Fixed SQL query: {fixed_sql}")

                # Try the fixed SQL
                try:
                    logger.info(f"Executing fixed SQL query: {fixed_sql}")
                    result, _ = mcp_client.query(fixed_sql)
                    logger.info(f"Fixed SQL query executed successfully")
                    # If successful, update the SQL for the interpretation step
                    sql = fixed_sql
                except Exception as e:
                    # If the fixed SQL also fails, try one more approach with explicit date column information
                    logger.error(f"Fixed SQL query failed with error: {str(e)}")
                    retry_prompt = f"""
The SQL query still has errors. Here are the available date/time columns in the database:

{datetime_context}

Original query:
```sql
{sql}
```

Error: {str(e)}

Please rewrite the query using one of these date/time columns. Only return the corrected SQL query.
"""
                    logger.info(f"Sending retry prompt to {selected_llm} model with date/time column information")
                    
                    if selected_llm in ["groq", "openai"]:
                        retry_response = llm_client.chat.completions.create(
                            messages=[
                                {"role": "user", "content": retry_prompt}],
                            model=model_name,
                            temperature=0,
                            max_tokens=1000,
                        )
                        retry_sql = retry_response.choices[0].message.content.strip()
                        logger.info(f"Raw retry response from {selected_llm}: {retry_sql[:200]}...")
                    else:  # Gemini
                        retry_response = llm_client.generate_content(
                            retry_prompt)
                        retry_sql = retry_response.text.strip()
                        logger.info(f"Raw retry response from {selected_llm}: {retry_sql[:200]}...")

                    # Extract SQL from markdown code blocks if present
                    sql_match = re.search(
                        r'```(?:sql)?\s*(.*?)\s*```', retry_sql, re.DOTALL)
                    if sql_match:
                        retry_sql = sql_match.group(1).strip()
                        logger.info(f"Extracted SQL from code block: {retry_sql}")
                    
                    logger.info(f"Retry SQL query: {retry_sql}")

                    try:
                        logger.info(f"Executing retry SQL query: {retry_sql}")
                        result, _ = mcp_client.query(retry_sql)
                        logger.info(f"Retry SQL query executed successfully")
                        # If successful, update the SQL for the interpretation step
                        sql = retry_sql
                    except Exception as e2:
                        # If all attempts fail, return the original error
                        logger.error(f"Retry SQL query failed with error: {str(e2)}")
                        return f"Tool execution failed: {str(e2)}\n\nOriginal SQL had issues: {error_message}"
            else:
                # If validation passed or we have a corrected version, use it
                if corrected_sql != sql:
                    sql = corrected_sql
                    logger.info(f"Using corrected SQL from validation: {sql}")

                logger.info(f"Executing validated SQL query: {sql}")
                result, _ = mcp_client.query(sql)
                logger.info(f"SQL query executed successfully")
        else:
            logger.error(f"Unsupported tool: {tool}")
            return f" Unsupported tool: {tool}"
    except Exception as e:
        logger.error(f"Tool execution failed: {str(e)}", exc_info=True)
        return f" Tool execution failed: {str(e)}"

    # Step 4: Send result back to AI to interpret
    logger.info(f"Tool execution successful, sending result to LLM for interpretation")
    interpretation_prompt = f"""
You are a smart AI that just used the MCP tool `{tool}`.

Now, interpret the result below and explain it to the user in simple, helpful language. 
"Use PostgreSQL-compatible SQL syntax. Avoid MySQL-only functions."
Don't show raw JSON. Format your response as follows:

1. For tabular data: Present the information in a well-formatted markdown table
2. For relationships: Clearly explain the relationships between tables
3. For query results: Summarize the data and highlight key insights
4. For errors: Explain what went wrong in simple terms

User Question:
{user_question}

Result:
{result}
"""

    try:
        logger.info(f"Sending interpretation prompt to {selected_llm} model {model_name}")
        
        if selected_llm in ["groq", "openai"]:
            response = llm_client.chat.completions.create(
                messages=[{"role": "user", "content": interpretation_prompt}],
                model=model_name,
                temperature=0.7,
                max_tokens=1000,
            )
            final_answer = response.choices[0].message.content.strip()
            logger.info(f"Interpretation response from {selected_llm}: {final_answer[:200]}...")
        else:  # Gemini
            response = llm_client.generate_content(interpretation_prompt)
            final_answer = response.text.strip()
            logger.info(f"Interpretation response from {selected_llm}: {final_answer[:200]}...")

            # Clean up any potential markdown formatting
            final_answer = re.sub(
                r'```.*?```', '', final_answer, flags=re.DOTALL)
            final_answer = re.sub(
                r'^[*#].*$', '', final_answer, flags=re.MULTILINE)
            final_answer = final_answer.strip()

        logger.info(f"Final answer prepared successfully")
        return final_answer
    except Exception as e:
        logger.error(f"Failed to interpret tool result: {str(e)}", exc_info=True)
        return f" Failed to interpret tool result: {e}"
