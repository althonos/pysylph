import os
import unittest
import importlib.resources
from contextlib import nullcontext

from pysylph import Database



class TestDatabase(unittest.TestCase):

    def test_init_empty(self):
        db1 = Database()
        self.assertEqual(len(db1), 0)
        db2 = Database([])
        self.assertEqual(len(db2), 0)

    def test_init_type_error(self):
        with self.assertRaises(TypeError):
            db = Database(123)

    def test_load_filename(self):
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(importlib.resources.files(__package__).joinpath("ecoli.syldb"))
        else:
            handler = importlib.resources.path(__package__, "ecoli.syldb")
        with handler as path:
            database = Database.load(os.fspath(path))
        self.assertEqual(len(database), 3)

    def test_load_path(self):
        if hasattr(importlib.resources, "files"):
            handler = nullcontext(importlib.resources.files(__package__).joinpath("ecoli.syldb"))
        else:
            handler = importlib.resources.path(__package__, "ecoli.syldb")
        with handler as path:
            database = Database.load(path)
        self.assertEqual(len(database), 3)

    def test_load_file_object(self):
        if hasattr(importlib.resources, "files"):
            handler = importlib.resources.files(__package__).joinpath("ecoli.syldb").open("rb")
        else:
            handler = importlib.resources.open_binary(__package__, "ecoli.syldb")
        with handler as f:
            database = Database.load(f)
        self.assertEqual(len(database), 3)