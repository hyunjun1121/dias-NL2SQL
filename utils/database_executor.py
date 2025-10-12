"""Database SQL executor."""

import sqlite3
import time
from typing import Dict, Any, Optional


class DatabaseExecutor:
    """Execute SQL on database."""

    def __init__(self, db_path: str, timeout: int = 30):
        self.db_path = db_path
        self.timeout = timeout

    def execute(self, sql: str) -> Dict[str, Any]:
        """Execute SQL and return result."""
        start_time = time.time()

        try:
            conn = sqlite3.connect(self.db_path, timeout=self.timeout)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()

            execution_time = time.time() - start_time

            return {
                'success': True,
                'result': result,
                'num_rows': len(result),
                'execution_time': execution_time,
                'error': None
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                'success': False,
                'result': None,
                'num_rows': 0,
                'execution_time': execution_time,
                'error': str(e)
            }
