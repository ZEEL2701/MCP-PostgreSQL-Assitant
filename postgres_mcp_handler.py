from typing import Optional, List
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
import logging
import decimal

logger = logging.getLogger("postgres-mcp-handler")

class PostgresMCPHandler:
    def __init__(self, conn_string: str):
        self.conn_string = conn_string
        self.pool = None
        self._init_pool()

    def _init_pool(self):
        try:
            self.pool = SimpleConnectionPool(1, 5, self.conn_string)
            logger.debug("Connection pool created successfully")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {str(e)}")
            self.pool = None

    def _get_connection(self):
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")
        return self.pool.getconn()

    def _release_connection(self, conn):
        if self.pool and conn:
            self.pool.putconn(conn)

    def _execute(self, sql: str, parameters: Optional[List] = None) -> List[dict]:
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, parameters)
                if cur.description:
                    rows = cur.fetchall()
                    safe_rows = []
                    for row in rows:
                            clean_row = {}
                            for key, value in row.items():
                                clean_row[key] = float(value) if isinstance(value, decimal.Decimal) else value
                            safe_rows.append(clean_row)
                    return safe_rows
                conn.commit()
                return [{"affected_rows": cur.rowcount}]
        finally:
            self._release_connection(conn)

    def health_check(self) -> List[dict]:
        return self._execute("SELECT 1 as healthy")

    def list_tables(self, db_schema: str = "public") -> List[dict]:
        sql = """
        SELECT table_name, table_type
        FROM information_schema.tables
        WHERE table_schema = %s
        ORDER BY table_name;
        """
        return self._execute(sql, [db_schema])

    def describe_table(self, table_name: str, db_schema: str = "public") -> List[dict]:
        sql = """
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position;
        """
        return self._execute(sql, [db_schema, table_name])

    def find_relationships(self, table_name: str, db_schema: str = "public") -> List[dict]:
        sql = """
        SELECT 
            kcu.column_name,
            ccu.table_name AS foreign_table,
            ccu.column_name AS foreign_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
            ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage ccu
            ON ccu.constraint_name = tc.constraint_name
        WHERE constraint_type = 'FOREIGN KEY'
            AND tc.table_schema = %s
            AND tc.table_name = %s;
        """
        return self._execute(sql, [db_schema, table_name])

    def query(self, sql: str, parameters: Optional[List] = None) -> List[dict]:
        return self._execute(sql, parameters)

    def close_pool(self):
        if self.pool:
            self.pool.closeall()
            logger.debug("Connection pool closed")
