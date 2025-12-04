from database import DatabaseManager
import unittest
import os
import shutil
import tempfile
from pathlib import Path
import sys

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / 'src'))


class TestDatabaseManager(unittest.TestCase):
    def setUp(self):
        # Use a temporary directory for testing
        self.test_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.test_dir.name) / 'test_lingkui.db'
        self.db = DatabaseManager(db_path=self.test_db_path)

    def tearDown(self):
        # Clean up the temporary directory
        self.test_dir.cleanup()

    def test_add_and_get_information(self):
        # Test adding information
        content = "The sky is blue."
        category = "fact"
        metadata = {"source": "observation"}

        row_id = self.db.add_information(content, category, metadata)
        self.assertIsNotNone(row_id)

        # Test retrieving information
        results = self.db.get_information(category="fact")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['content'], content)
        self.assertEqual(results[0]['category'], category)
        self.assertEqual(results[0]['metadata'], metadata)

    def test_search_information(self):
        self.db.add_information("Python is a programming language.", "tech")
        self.db.add_information("Snakes are reptiles.", "biology")

        # Search for "Python"
        results = self.db.search_information("Python")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['content'],
                         "Python is a programming language.")

        # Search for something that doesn't exist
        results = self.db.search_information("Java")
        self.assertEqual(len(results), 0)

    def test_get_information_limit(self):
        for i in range(5):
            self.db.add_information(f"Info {i}", "test")

        results = self.db.get_information(category="test", limit=3)
        self.assertEqual(len(results), 3)


if __name__ == '__main__':
    unittest.main()
