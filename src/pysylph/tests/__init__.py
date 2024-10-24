from . import (
    test_database,
    test_doctest,
    test_sketcher,
    test_profiler
)

def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(test_database))
    suite.addTests(loader.loadTestsFromModule(test_doctest))
    suite.addTests(loader.loadTestsFromModule(test_sketcher))
    suite.addTests(loader.loadTestsFromModule(test_profiler))
    return suite
