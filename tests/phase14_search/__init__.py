"""Phase 14 Search Tests - Real integration tests for search functionality.

This module contains integration tests for the search system with NO MOCK THEATER:
- All tests use real files written to temp directories
- All tests verify actual search behavior
- All tests run against real backends (text, pattern, semantic)

Test files:
- test_text_search.py: Text/substring search with real files
- test_pattern_search.py: Regex and AST pattern matching
- test_search_service.py: Unified SearchService integration
- test_cli_search.py: CLI command integration tests
- conftest.py: Real fixtures with temp directories

Key principles:
1. NO MOCKS - Tests use real file I/O and real search backends
2. REAL BEHAVIOR - Tests verify actual search results, not mock calls
3. INTEGRATION - Tests cover the full search path from API to results
"""
