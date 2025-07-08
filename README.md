# MCP-PostgreSQL-Assistant

An MCP server for PostgreSQL that generates questions and uses tools to answer questions about your database.

## Features

- Connect to any PostgreSQL database
- Choose between Groq, OpenAI, or Gemini as your LLM provider
- Automatically analyze database schema
- Generate intelligent questions based on your database structure
- Execute SQL queries through natural language
- View table relationships and structure
- Interactive chat interface

## Setup

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   streamlit run app.py
   ```

## Configuration

1. Obtain API keys:
   - Groq API key: https://console.groq.com/
   - OpenAI API key: https://platform.openai.com/
   - Gemini API key: https://ai.google.dev/

2. Enter your database credentials:
   - Host
   - Port
   - Database name
   - Username
   - Password

## Usage

1. Select your preferred LLM provider (Groq, OpenAI, or Gemini)
2. Enter your API key
3. Connect to your PostgreSQL database
4. The application will analyze your schema and generate relevant questions
5. Select a question or type your own
6. View the answer and explore your data

## Advanced Tools

- List tables
- Describe table structure
- Find relationships between tables
- Execute custom SQL queries

## Troubleshooting

If you're experiencing low accuracy with queries:
- Try switching between Groq, OpenAI, and Gemini to see which works better for your use case
- Make sure your database schema is properly analyzed
- Check that foreign key relationships are properly defined in your database
- Use more specific questions that reference actual table and column names

## License

See LICENSE file for details.
