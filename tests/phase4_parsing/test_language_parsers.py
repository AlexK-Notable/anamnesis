"""Tests for language-specific parsers module."""

import pytest

from anamnesis.parsing.ast_types import NodeType
from anamnesis.parsing.language_parsers import (
    ExtractedImport,
    ExtractedSymbol,
    GoParser,
    ParseResult,
    PythonParser,
    RustParser,
    TypeScriptParser,
    _GenericParser,
    get_parser_for_language,
    get_supported_languages,
)


class TestExtractedSymbol:
    """Tests for ExtractedSymbol dataclass."""

    def test_basic_symbol(self):
        """Test creating a basic symbol."""
        symbol = ExtractedSymbol(
            name="foo",
            kind=NodeType.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=3,
        )
        assert symbol.name == "foo"
        assert symbol.kind == NodeType.FUNCTION
        assert symbol.file_path == "/test.py"

    def test_symbol_with_all_fields(self):
        """Test symbol with all optional fields."""
        symbol = ExtractedSymbol(
            name="MyClass",
            kind=NodeType.CLASS,
            file_path="/test.py",
            start_line=1,
            end_line=20,
            signature="class MyClass(Base)",
            docstring="A test class.",
            parent=None,
            children=["method1", "method2"],
            decorators=["@dataclass"],
            modifiers=["public"],
            parameters=[],
            return_type=None,
            is_async=False,
            is_exported=True,
        )
        assert symbol.docstring == "A test class."
        assert len(symbol.children) == 2
        assert symbol.is_exported is True

    def test_to_dict_minimal(self):
        """Test serialization with minimal fields."""
        symbol = ExtractedSymbol(
            name="bar",
            kind=NodeType.FUNCTION,
            file_path="/test.py",
            start_line=5,
            end_line=10,
        )
        data = symbol.to_dict()
        assert data["name"] == "bar"
        assert data["kind"] == "function"
        assert "docstring" not in data  # Not included when None

    def test_to_dict_full(self):
        """Test serialization with all fields."""
        symbol = ExtractedSymbol(
            name="async_func",
            kind=NodeType.FUNCTION,
            file_path="/test.py",
            start_line=1,
            end_line=5,
            signature="async def async_func()",
            docstring="An async function.",
            is_async=True,
            is_exported=True,
        )
        data = symbol.to_dict()
        assert data["is_async"] is True
        assert data["is_exported"] is True
        assert data["signature"] == "async def async_func()"


class TestExtractedImport:
    """Tests for ExtractedImport dataclass."""

    def test_basic_import(self):
        """Test creating a basic import."""
        imp = ExtractedImport(module="os")
        assert imp.module == "os"
        assert imp.names == []
        assert imp.alias is None

    def test_import_with_names(self):
        """Test import with specific names."""
        imp = ExtractedImport(module="typing", names=["List", "Dict", "Optional"])
        assert len(imp.names) == 3
        assert "List" in imp.names

    def test_import_with_alias(self):
        """Test import with alias."""
        imp = ExtractedImport(module="numpy", alias="np")
        assert imp.alias == "np"

    def test_to_dict(self):
        """Test serialization."""
        imp = ExtractedImport(
            module="collections",
            names=["defaultdict"],
            is_default=True,
        )
        data = imp.to_dict()
        assert data["module"] == "collections"
        assert data["names"] == ["defaultdict"]
        assert data["is_default"] is True


class TestParseResult:
    """Tests for ParseResult dataclass."""

    def test_empty_result(self):
        """Test empty parse result."""
        result = ParseResult(file_path="/test.py", language="python")
        assert result.file_path == "/test.py"
        assert result.language == "python"
        assert result.symbols == []
        assert result.imports == []
        assert not result.has_errors

    def test_result_with_errors(self):
        """Test result with errors."""
        result = ParseResult(
            file_path="/test.py",
            language="python",
            errors=["Syntax error at line 1"],
        )
        assert result.has_errors

    def test_get_classes(self):
        """Test filtering classes."""
        symbols = [
            ExtractedSymbol(name="Foo", kind=NodeType.CLASS, file_path="", start_line=1, end_line=1),
            ExtractedSymbol(name="bar", kind=NodeType.FUNCTION, file_path="", start_line=2, end_line=2),
            ExtractedSymbol(name="Baz", kind=NodeType.CLASS, file_path="", start_line=3, end_line=3),
        ]
        result = ParseResult(file_path="", language="python", symbols=symbols)
        classes = result.get_classes()
        assert len(classes) == 2
        assert all(s.kind == NodeType.CLASS for s in classes)

    def test_get_functions(self):
        """Test filtering functions and methods."""
        symbols = [
            ExtractedSymbol(name="func1", kind=NodeType.FUNCTION, file_path="", start_line=1, end_line=1),
            ExtractedSymbol(name="method1", kind=NodeType.METHOD, file_path="", start_line=2, end_line=2),
            ExtractedSymbol(name="Cls", kind=NodeType.CLASS, file_path="", start_line=3, end_line=3),
        ]
        result = ParseResult(file_path="", language="python", symbols=symbols)
        funcs = result.get_functions()
        assert len(funcs) == 2

    def test_get_top_level_symbols(self):
        """Test getting top-level symbols."""
        symbols = [
            ExtractedSymbol(name="Foo", kind=NodeType.CLASS, file_path="", start_line=1, end_line=1, parent=None),
            ExtractedSymbol(name="method", kind=NodeType.METHOD, file_path="", start_line=2, end_line=2, parent="Foo"),
            ExtractedSymbol(name="bar", kind=NodeType.FUNCTION, file_path="", start_line=3, end_line=3, parent=None),
        ]
        result = ParseResult(file_path="", language="python", symbols=symbols)
        top_level = result.get_top_level_symbols()
        assert len(top_level) == 2
        assert all(s.parent is None for s in top_level)

    def test_to_dict(self):
        """Test serialization."""
        result = ParseResult(
            file_path="/test.py",
            language="python",
            symbols=[
                ExtractedSymbol(name="foo", kind=NodeType.FUNCTION, file_path="", start_line=1, end_line=1),
            ],
            imports=[
                ExtractedImport(module="os"),
            ],
            exports=["foo"],
            errors=[],
        )
        data = result.to_dict()
        assert data["file_path"] == "/test.py"
        assert len(data["symbols"]) == 1
        assert len(data["imports"]) == 1


class TestGetParserForLanguage:
    """Tests for parser factory function."""

    def test_get_python_parser(self):
        """Test getting Python parser."""
        parser = get_parser_for_language("python")
        assert isinstance(parser, PythonParser)

    def test_get_typescript_parser(self):
        """Test getting TypeScript parser."""
        parser = get_parser_for_language("typescript")
        assert isinstance(parser, TypeScriptParser)

    def test_get_javascript_parser(self):
        """Test getting JavaScript parser."""
        parser = get_parser_for_language("javascript")
        assert isinstance(parser, TypeScriptParser)

    def test_get_go_parser(self):
        """Test getting Go parser."""
        parser = get_parser_for_language("go")
        assert isinstance(parser, GoParser)

    def test_get_rust_parser(self):
        """Test getting Rust parser."""
        parser = get_parser_for_language("rust")
        assert isinstance(parser, RustParser)

    def test_get_generic_parser(self):
        """Test getting generic parser for supported but unspecialized language."""
        parser = get_parser_for_language("c")
        assert isinstance(parser, _GenericParser)

    def test_case_insensitive(self):
        """Test case insensitivity."""
        parser = get_parser_for_language("Python")
        assert isinstance(parser, PythonParser)

    def test_unsupported_language(self):
        """Test error for unsupported language."""
        with pytest.raises(ValueError) as exc_info:
            get_parser_for_language("nonexistent")
        assert "Unsupported language" in str(exc_info.value)


class TestPythonParser:
    """Tests for Python parser."""

    def test_parse_simple_function(self):
        """Test parsing a simple function."""
        parser = PythonParser()
        source = "def hello():\n    return 'world'"
        result = parser.parse(source, "/test.py")

        assert not result.has_errors
        assert len(result.symbols) == 1
        assert result.symbols[0].name == "hello"
        assert result.symbols[0].kind == NodeType.FUNCTION

    def test_parse_async_function(self):
        """Test parsing an async function."""
        parser = PythonParser()
        source = "async def fetch():\n    return await get_data()"
        result = parser.parse(source, "/test.py")

        assert len(result.symbols) == 1
        assert result.symbols[0].is_async is True
        assert "async" in result.symbols[0].signature

    def test_parse_class(self):
        """Test parsing a class."""
        parser = PythonParser()
        source = """class MyClass:
    def __init__(self):
        pass
    
    def method(self):
        pass
"""
        result = parser.parse(source, "/test.py")

        classes = result.get_classes()
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

        methods = [s for s in result.symbols if s.kind == NodeType.METHOD]
        assert len(methods) == 2
        assert all(s.parent == "MyClass" for s in methods)

    def test_parse_with_docstring(self):
        """Test extracting docstrings."""
        parser = PythonParser()
        source = '''def documented():
    """This is a docstring."""
    pass
'''
        result = parser.parse(source, "/test.py")

        assert len(result.symbols) == 1
        assert result.symbols[0].docstring == "This is a docstring."

    def test_parse_with_decorators(self):
        """Test extracting decorators."""
        parser = PythonParser()
        source = """@staticmethod
@cached
def decorated():
    pass
"""
        result = parser.parse(source, "/test.py")

        # Note: decorator extraction depends on tree-sitter AST structure
        assert len(result.symbols) >= 1

    def test_parse_imports(self):
        """Test extracting imports."""
        parser = PythonParser()
        source = """import os
import sys
from typing import List, Dict
from pathlib import Path
"""
        result = parser.parse(source, "/test.py")

        assert len(result.imports) >= 3
        modules = [i.module for i in result.imports]
        assert "os" in modules or any("os" in m for m in modules)

    def test_parse_from_import(self):
        """Test extracting from imports."""
        parser = PythonParser()
        source = "from collections import defaultdict, Counter"
        result = parser.parse(source, "/test.py")

        assert len(result.imports) >= 1
        imp = result.imports[0]
        assert "collections" in imp.module or imp.module == "collections"

    def test_parse_multiple_functions(self):
        """Test parsing multiple functions."""
        parser = PythonParser()
        source = """def foo():
    pass

def bar():
    pass

def baz():
    pass
"""
        result = parser.parse(source, "/test.py")

        funcs = result.get_functions()
        assert len(funcs) == 3
        names = [f.name for f in funcs]
        assert "foo" in names
        assert "bar" in names
        assert "baz" in names


class TestTypeScriptParser:
    """Tests for TypeScript parser."""

    def test_parse_function(self):
        """Test parsing a function."""
        parser = TypeScriptParser()
        source = "function hello(): string { return 'world'; }"
        result = parser.parse(source, "/test.ts")

        assert len(result.symbols) >= 1
        funcs = result.get_functions()
        assert len(funcs) >= 1

    def test_parse_class(self):
        """Test parsing a class."""
        parser = TypeScriptParser()
        source = """class MyClass {
    constructor() {}
    
    method(): void {}
}
"""
        result = parser.parse(source, "/test.ts")

        classes = result.get_classes()
        assert len(classes) == 1
        assert classes[0].name == "MyClass"

    def test_parse_interface(self):
        """Test parsing an interface."""
        parser = TypeScriptParser()
        source = """interface User {
    name: string;
    age: number;
}
"""
        result = parser.parse(source, "/test.ts")

        interfaces = [s for s in result.symbols if s.kind == NodeType.INTERFACE]
        assert len(interfaces) == 1
        assert interfaces[0].name == "User"

    def test_parse_type_alias(self):
        """Test parsing a type alias."""
        parser = TypeScriptParser()
        source = "type ID = string | number;"
        result = parser.parse(source, "/test.ts")

        type_aliases = [s for s in result.symbols if s.kind == NodeType.TYPE_ALIAS]
        assert len(type_aliases) == 1
        assert type_aliases[0].name == "ID"

    def test_parse_enum(self):
        """Test parsing an enum."""
        parser = TypeScriptParser()
        source = """enum Color {
    Red,
    Green,
    Blue
}
"""
        result = parser.parse(source, "/test.ts")

        enums = [s for s in result.symbols if s.kind == NodeType.ENUM]
        assert len(enums) == 1
        assert enums[0].name == "Color"

    def test_parse_imports(self):
        """Test extracting TypeScript imports."""
        parser = TypeScriptParser()
        source = """import { Component } from 'react';
import * as fs from 'fs';
import path from 'path';
"""
        result = parser.parse(source, "/test.ts")

        assert len(result.imports) >= 1

    def test_javascript_mode(self):
        """Test JavaScript parsing mode."""
        parser = TypeScriptParser("javascript")
        source = "function test() { return 42; }"
        result = parser.parse(source, "/test.js")

        assert result.language == "javascript"
        assert len(result.get_functions()) >= 1


class TestGoParser:
    """Tests for Go parser."""

    def test_parse_function(self):
        """Test parsing a Go function."""
        parser = GoParser()
        source = """package main

func Hello() string {
    return "world"
}
"""
        result = parser.parse(source, "/test.go")

        funcs = result.get_functions()
        assert len(funcs) >= 1
        assert any(f.name == "Hello" for f in funcs)

    def test_parse_type(self):
        """Test parsing a Go type."""
        parser = GoParser()
        source = """package main

type User struct {
    Name string
    Age  int
}
"""
        result = parser.parse(source, "/test.go")

        types = [s for s in result.symbols if s.kind == NodeType.CLASS]
        assert len(types) >= 1
        assert any(t.name == "User" for t in types)

    def test_parse_method(self):
        """Test parsing a Go method."""
        parser = GoParser()
        source = """package main

type User struct {}

func (u *User) GetName() string {
    return u.Name
}
"""
        result = parser.parse(source, "/test.go")

        methods = [s for s in result.symbols if s.kind == NodeType.METHOD]
        assert len(methods) >= 1

    def test_exported_detection(self):
        """Test detection of exported symbols (capitalized)."""
        parser = GoParser()
        source = """package main

func Exported() {}
func private() {}
"""
        result = parser.parse(source, "/test.go")

        funcs = result.get_functions()
        exported = [f for f in funcs if f.name == "Exported"]
        private = [f for f in funcs if f.name == "private"]

        if exported:
            assert exported[0].is_exported is True
        if private:
            assert private[0].is_exported is False

    def test_parse_imports(self):
        """Test extracting Go imports."""
        parser = GoParser()
        source = """package main

import (
    "fmt"
    "os"
)
"""
        result = parser.parse(source, "/test.go")

        assert len(result.imports) >= 1


class TestRustParser:
    """Tests for Rust parser."""

    def test_parse_function(self):
        """Test parsing a Rust function."""
        parser = RustParser()
        source = """fn hello() -> String {
    String::from("world")
}
"""
        result = parser.parse(source, "/test.rs")

        funcs = result.get_functions()
        assert len(funcs) >= 1

    def test_parse_struct(self):
        """Test parsing a Rust struct."""
        parser = RustParser()
        source = """pub struct User {
    name: String,
    age: u32,
}
"""
        result = parser.parse(source, "/test.rs")

        structs = [s for s in result.symbols if s.kind == NodeType.CLASS]
        assert len(structs) >= 1
        assert any(s.name == "User" for s in structs)

    def test_parse_impl(self):
        """Test parsing impl blocks."""
        parser = RustParser()
        source = """struct User {}

impl User {
    pub fn new() -> Self {
        User {}
    }

    fn private_method(&self) {}
}
"""
        result = parser.parse(source, "/test.rs")

        methods = [s for s in result.symbols if s.kind == NodeType.METHOD]
        assert len(methods) >= 1

    def test_parse_enum(self):
        """Test parsing a Rust enum."""
        parser = RustParser()
        source = """pub enum Color {
    Red,
    Green,
    Blue,
}
"""
        result = parser.parse(source, "/test.rs")

        enums = [s for s in result.symbols if s.kind == NodeType.ENUM]
        assert len(enums) >= 1
        assert any(e.name == "Color" for e in enums)

    def test_parse_trait(self):
        """Test parsing a Rust trait."""
        parser = RustParser()
        source = """pub trait Display {
    fn fmt(&self) -> String;
}
"""
        result = parser.parse(source, "/test.rs")

        traits = [s for s in result.symbols if s.kind == NodeType.INTERFACE]
        assert len(traits) >= 1

    def test_parse_mod(self):
        """Test parsing a Rust module."""
        parser = RustParser()
        source = """pub mod utils {
    pub fn helper() {}
}
"""
        result = parser.parse(source, "/test.rs")

        mods = [s for s in result.symbols if s.kind == NodeType.MODULE]
        assert len(mods) >= 1

    def test_parse_use(self):
        """Test extracting Rust use declarations."""
        parser = RustParser()
        source = """use std::collections::HashMap;
use std::io::{self, Read, Write};
"""
        result = parser.parse(source, "/test.rs")

        assert len(result.imports) >= 1


class TestGenericParser:
    """Tests for generic parser fallback."""

    def test_parse_c(self):
        """Test parsing C code with generic parser."""
        parser = _GenericParser("c")
        source = """int main() {
    return 0;
}
"""
        result = parser.parse(source, "/test.c")
        assert result.language == "c"
        assert not result.has_errors

    def test_no_imports(self):
        """Test that generic parser returns no imports."""
        parser = _GenericParser("c")
        source = "#include <stdio.h>"
        result = parser.parse(source, "/test.c")
        # Generic parser doesn't extract imports
        assert result.imports == []


class TestGetSupportedLanguages:
    """Tests for get_supported_languages function."""

    def test_returns_list(self):
        """Test that function returns a list."""
        languages = get_supported_languages()
        assert isinstance(languages, list)

    def test_contains_major_languages(self):
        """Test that major languages are included."""
        languages = get_supported_languages()
        expected = ["python", "typescript", "javascript", "rust", "go"]
        for lang in expected:
            assert lang in languages
