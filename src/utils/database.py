import sqlite3
import json
from datetime import datetime
from pathlib import Path


class DatabaseManager:
    def __init__(self, db_path=None):
        if db_path is None:
            # Default to data/lingkui.db in the project root
            project_root = Path(__file__).parent.parent
            self.db_path = project_root / 'data' / 'lingkui.db'
        else:
            self.db_path = Path(db_path)

        self._init_db()

    def _get_connection(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize the database with required tables."""
        # Ensure the parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Create a general purpose information table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS information (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                category TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')

        conn.commit()
        conn.close()

    def add_information(self, content, category=None, metadata=None):
        """
        Add a new piece of information to the database.

        Args:
            content (str): The main content/information.
            category (str, optional): A category tag for the information.
            metadata (dict, optional): Additional metadata as a dictionary.

        Returns:
            int: The ID of the inserted row.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        metadata_json = json.dumps(metadata) if metadata else None

        cursor.execute('''
            INSERT INTO information (content, category, metadata)
            VALUES (?, ?, ?)
        ''', (content, category, metadata_json))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return row_id

    def get_information(self, category=None, limit=10):
        """
        Retrieve information from the database.

        Args:
            category (str, optional): Filter by category.
            limit (int): Maximum number of records to return.

        Returns:
            list: List of dictionaries containing the information.
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = 'SELECT * FROM information'
        params = []

        if category:
            query += ' WHERE category = ?'
            params.append(category)

        query += ' ORDER BY created_at DESC LIMIT ?'
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            item = dict(row)
            if item['metadata']:
                item['metadata'] = json.loads(item['metadata'])
            results.append(item)

        conn.close()
        return results

    def search_information(self, query_text):
        """
        Simple text search (LIKE query).

        Args:
            query_text (str): Text to search for in content.

        Returns:
            list: List of matching records.
        """
        conn = self._get_connection()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        sql = 'SELECT * FROM information WHERE content LIKE ? ORDER BY created_at DESC'
        cursor.execute(sql, (f'%{query_text}%',))

        rows = cursor.fetchall()
        results = []
        for row in rows:
            item = dict(row)
            if item['metadata']:
                item['metadata'] = json.loads(item['metadata'])
            results.append(item)

        conn.close()
        return results


# Global instance
db = DatabaseManager()
