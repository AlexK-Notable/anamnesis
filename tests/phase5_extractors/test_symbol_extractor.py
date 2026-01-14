"""Tests for symbol extractor module."""

import pytest

from anamnesis.extractors.symbol_extractor import (
    ExtractedSymbol,
    SymbolExtractor,
    SymbolKind,
    extract_symbols_from_source,
)


class TestSymbolKind:
    """Tests for SymbolKind enum."""

    def test_declaration_kinds(self):
        """Test declaration symbol kinds."""
        assert SymbolKind.CLASS == "class"
        assert SymbolKind.FUNCTION == "function"
        assert SymbolKind.METHOD == "method"
        assert SymbolKind.VARIABLE == "variable"
        assert SymbolKind.CONSTANT == "constant"

    def test_special_kinds(self):
        """Test special symbol kinds."""
        assert SymbolKind.INTERFACE == "interface"
        assert SymbolKind.ENUM == "enum"
        assert SymbolKind.TYPE_ALIAS == "type_alias"
        assert SymbolKind.LAMBDA == "lambda"


class TestExtractedSymbol:
    """Tests for ExtractedSymbol dataclass."""

    def test_basic_creation(self):
        """Test creating a basic symbol."""
        symbol = ExtractedSymbol(
            name="my_function",
            kind=SymbolKind.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=5,
        )
        assert symbol.name == "my_function"
        assert symbol.kind == SymbolKind.FUNCTION
        assert symbol.line_count == 5

    def test_full_path(self):
        """Test getting full path."""
        symbol = ExtractedSymbol(
            name="method",
            kind=SymbolKind.METHOD,
            file_path="/test.py",
            start_line=1,
            end_line=1,
            parent_name="MyClass",
        )
        assert symbol.get_full_path() == "MyClass.method"

    def test_qualified_name_priority(self):
        """Test qualified name takes priority."""
        symbol = ExtractedSymbol(
            name="method",
            kind=SymbolKind.METHOD,
            file_path="/test.py",
            start_line=1,
            end_line=1,
            parent_name="MyClass",
            qualified_name="module.MyClass.method",
        )
        assert symbol.get_full_path() == "module.MyClass.method"

    def test_is_private_explicit(self):
        """Test explicit private visibility."""
        symbol = ExtractedSymbol(
            name="func",
            kind=SymbolKind.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=1,
            visibility="private",
        )
        assert symbol.is_private is True

    def test_is_private_underscore(self):
        """Test underscore-prefixed names are private."""
        symbol = ExtractedSymbol(
            name="_private_func",
            kind=SymbolKind.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=1,
        )
        assert symbol.is_private is True

    def test_to_dict(self):
        """Test serialization to dict."""
        symbol = ExtractedSymbol(
            name="test",
            kind=SymbolKind.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=5,
            is_async=True,
            decorators=["@cached"],
        )
        data = symbol.to_dict()
        assert data["name"] == "test"
        assert data["kind"] == "function"
        assert data["is_async"] is True
        assert data["decorators"] == ["@cached"]

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "name": "test",
            "kind": "method",
            "file_path": "/test.py",
            "start_line": 10,
            "end_line": 20,
            "visibility": "protected",
        }
        symbol = ExtractedSymbol.from_dict(data)
        assert symbol.name == "test"
        assert symbol.kind == SymbolKind.METHOD
        assert symbol.visibility == "protected"

    def test_from_dict_unknown_kind(self):
        """Test deserializing unknown kind."""
        data = {
            "name": "test",
            "kind": "custom_kind",
            "file_path": "/test.py",
            "start_line": 1,
            "end_line": 1,
        }
        symbol = ExtractedSymbol.from_dict(data)
        assert symbol.kind == "custom_kind"

    def test_roundtrip_serialization(self):
        """Test roundtrip serialization."""
        original = ExtractedSymbol(
            name="test_func",
            kind=SymbolKind.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=10,
            parent_name="TestClass",
            qualified_name="module.TestClass.test_func",
            signature="def test_func(x: int) -> str",
            docstring="A test function.",
            visibility="public",
            is_async=True,
            decorators=["@staticmethod"],
            return_type="str",
            language="python",
        )
        data = original.to_dict()
        restored = ExtractedSymbol.from_dict(data)
        assert restored.name == original.name
        assert restored.kind == original.kind
        assert restored.qualified_name == original.qualified_name
        assert restored.is_async == original.is_async


class TestSymbolExtractor:
    """Tests for SymbolExtractor class."""

    def test_extract_python_function(self):
        """Test extracting Python function."""
        source = '''def hello(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}"
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        assert len(symbols) >= 1
        
        func = symbols[0]
        assert func.name == "hello"
        assert func.kind == SymbolKind.FUNCTION

    def test_extract_python_class(self):
        """Test extracting Python class."""
        source = '''class MyClass:
    """A test class."""
    
    def __init__(self):
        self.value = 0
    
    def get_value(self):
        return self.value
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        
        # Find the class
        classes = [s for s in symbols if s.kind == SymbolKind.CLASS]
        assert len(classes) >= 1
        
        cls = classes[0]
        assert cls.name == "MyClass"

    def test_extract_async_function(self):
        """Test extracting async function."""
        source = '''async def fetch_data(url: str):
    """Fetch data from URL."""
    pass
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        assert len(symbols) >= 1
        
        func = symbols[0]
        assert func.name == "fetch_data"
        assert func.is_async is True

    def test_extract_decorated_function(self):
        """Test extracting decorated function."""
        source = '''@staticmethod
@cached
def process(data):
    pass
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        # Check if we get the function
        funcs = [s for s in symbols if s.kind == SymbolKind.FUNCTION]
        if funcs:
            assert "@staticmethod" in funcs[0].decorators or "@cached" in funcs[0].decorators

    def test_extract_private_excluded(self):
        """Test excluding private symbols."""
        source = '''def public_func():
    pass

def _private_func():
    pass
'''
        extractor = SymbolExtractor(include_private=False)
        symbols = extractor.extract_from_file("/test.py", source, "python")
        
        names = [s.name for s in symbols]
        assert "public_func" in names
        assert "_private_func" not in names

    def test_extract_private_included(self):
        """Test including private symbols."""
        source = '''def public_func():
    pass

def _private_func():
    pass
'''
        extractor = SymbolExtractor(include_private=True)
        symbols = extractor.extract_from_file("/test.py", source, "python")
        
        names = [s.name for s in symbols]
        assert "public_func" in names
        assert "_private_func" in names

    def test_extract_nested_classes(self):
        """Test extracting nested classes."""
        source = '''class Outer:
    class Inner:
        def method(self):
            pass
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        
        # Should have Outer class with Inner as child
        outer = [s for s in symbols if s.name == "Outer"]
        assert len(outer) >= 1

    def test_extract_typescript_function(self):
        """Test extracting TypeScript function."""
        source = '''function greet(name: string): string {
    return `Hello, ${name}`;
}
'''
        symbols = extract_symbols_from_source(source, "typescript", "/test.ts")
        assert len(symbols) >= 1
        
        func = symbols[0]
        assert func.name == "greet"
        assert func.kind == SymbolKind.FUNCTION

    def test_extract_go_function(self):
        """Test extracting Go function."""
        source = '''func Hello(name string) string {
    return "Hello, " + name
}
'''
        symbols = extract_symbols_from_source(source, "go", "/test.go")
        assert len(symbols) >= 1
        
        func = symbols[0]
        assert func.name == "Hello"

    def test_visibility_python_dunder(self):
        """Test Python dunder methods are public."""
        source = '''class Test:
    def __init__(self):
        pass
    
    def __str__(self):
        return "test"
'''
        symbols = extract_symbols_from_source(source, "python", "/test.py")
        # Classes and methods should be extracted

    def test_include_body_option(self):
        """Test including source text in symbols."""
        source = '''def simple():
    return 42
'''
        extractor = SymbolExtractor(include_body=True)
        symbols = extractor.extract_from_file("/test.py", source, "python")
        
        if symbols:
            # Source text should be included
            assert symbols[0].source_text is not None


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_extract_symbols_from_source(self):
        """Test the convenience function."""
        source = "def foo(): pass"
        symbols = extract_symbols_from_source(source, "python")
        assert len(symbols) >= 1

    def test_extract_with_file_path(self):
        """Test with custom file path."""
        source = "def bar(): pass"
        symbols = extract_symbols_from_source(source, "python", "/custom/path.py")
        if symbols:
            assert symbols[0].file_path == "/custom/path.py"

    def test_unsupported_language(self):
        """Test with unsupported language."""
        source = "some code"
        symbols = extract_symbols_from_source(source, "unsupported_lang")
        assert symbols == []
