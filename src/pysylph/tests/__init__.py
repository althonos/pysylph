from . import (
    test_database,
    test_doctest
)

def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_database))
    suite.addTests(loader.loadTestsFromModule(test_doctest))
    return suite
