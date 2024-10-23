from . import (
    test_database
)

def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_database))
    return suite
