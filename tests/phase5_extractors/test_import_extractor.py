"""Tests for import extractor module."""


from anamnesis.extractors.import_extractor import (
    ExtractedImport,
    ImportedName,
    ImportExtractor,
    ImportKind,
    extract_imports_from_source,
)


class TestImportKind:
    """Tests for ImportKind enum."""

    def test_standard_imports(self):
        """Test standard import kinds."""
        assert ImportKind.IMPORT == "import"
        assert ImportKind.FROM_IMPORT == "from_import"
        assert ImportKind.IMPORT_ALIAS == "import_alias"
        assert ImportKind.STAR_IMPORT == "star_import"

    def test_special_imports(self):
        """Test special import kinds."""
        assert ImportKind.RELATIVE == "relative"
        assert ImportKind.DYNAMIC == "dynamic"
        assert ImportKind.TYPE_ONLY == "type_only"


class TestImportedName:
    """Tests for ImportedName dataclass."""

    def test_basic_name(self):
        """Test basic imported name."""
        name = ImportedName(name="foo")
        assert name.name == "foo"
        assert name.alias is None
        assert name.local_name == "foo"

    def test_aliased_name(self):
        """Test aliased imported name."""
        name = ImportedName(name="foo", alias="bar")
        assert name.name == "foo"
        assert name.alias == "bar"
        assert name.local_name == "bar"


class TestExtractedImport:
    """Tests for ExtractedImport dataclass."""

    def test_basic_creation(self):
        """Test creating a basic import."""
        imp = ExtractedImport(
            module="os",
            kind=ImportKind.IMPORT,
            file_path="/test.py",
            start_line=1,
            end_line=1,
        )
        assert imp.module == "os"
        assert imp.kind == ImportKind.IMPORT

    def test_imported_names(self):
        """Test getting imported names."""
        imp = ExtractedImport(
            module="typing",
            kind=ImportKind.FROM_IMPORT,
            file_path="/test.py",
            start_line=1,
            end_line=1,
            names=[
                ImportedName(name="List"),
                ImportedName(name="Dict", alias="D"),
            ],
        )
        assert imp.imported_names == ["List", "Dict"]
        assert imp.local_names == ["List", "D"]

    def test_to_dict(self):
        """Test serialization to dict."""
        imp = ExtractedImport(
            module="collections",
            kind=ImportKind.FROM_IMPORT,
            file_path="/test.py",
            start_line=1,
            end_line=1,
            names=[ImportedName(name="Counter")],
            is_stdlib=True,
        )
        data = imp.to_dict()
        assert data["module"] == "collections"
        assert data["kind"] == "from_import"
        assert data["is_stdlib"] is True

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "module": "os.path",
            "kind": "import",
            "file_path": "/test.py",
            "start_line": 1,
            "end_line": 1,
            "is_relative": False,
        }
        imp = ExtractedImport.from_dict(data)
        assert imp.module == "os.path"
        assert imp.kind == ImportKind.IMPORT

    def test_relative_import(self):
        """Test relative import properties."""
        imp = ExtractedImport(
            module="sibling",
            kind=ImportKind.RELATIVE,
            file_path="/test.py",
            start_line=1,
            end_line=1,
            is_relative=True,
            relative_level=1,
        )
        assert imp.is_relative is True
        assert imp.relative_level == 1


class TestImportExtractor:
    """Tests for ImportExtractor class."""

    def test_extract_simple_import(self):
        """Test extracting simple import."""
        source = "import os"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.module == "os"
        assert imp.kind == ImportKind.IMPORT

    def test_extract_from_import(self):
        """Test extracting from import."""
        source = "from collections import Counter"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.module == "collections"
        assert imp.kind == ImportKind.FROM_IMPORT
        assert "Counter" in [n.name for n in imp.names]

    def test_extract_aliased_import(self):
        """Test extracting aliased import."""
        source = "import numpy as np"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.module == "numpy"
        assert imp.kind == ImportKind.IMPORT_ALIAS

    def test_extract_relative_import(self):
        """Test extracting relative import."""
        source = "from . import sibling"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.is_relative is True
        assert imp.relative_level >= 1

    def test_extract_parent_relative_import(self):
        """Test extracting parent relative import."""
        source = "from .. import parent_module"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.is_relative is True
        assert imp.relative_level >= 2

    def test_extract_star_import(self):
        """Test extracting star import."""
        source = "from module import *"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.kind == ImportKind.STAR_IMPORT

    def test_extract_multiple_imports(self):
        """Test extracting multiple imports."""
        source = '''import os
import sys
from pathlib import Path
'''
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 3
        modules = [i.module for i in imports]
        assert "os" in modules
        assert "sys" in modules
        assert "pathlib" in modules

    def test_stdlib_detection(self):
        """Test standard library detection."""
        source = "import os"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        assert len(imports) >= 1
        assert imports[0].is_stdlib is True

    def test_non_stdlib_detection(self):
        """Test non-stdlib detection."""
        source = "import some_third_party_package"
        imports = extract_imports_from_source(source, "python", "/test.py")
        
        if imports:
            assert imports[0].is_stdlib is False

    def test_extract_typescript_import(self):
        """Test extracting TypeScript import."""
        source = "import { Component } from 'react';"
        imports = extract_imports_from_source(source, "typescript", "/test.ts")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert "react" in imp.module

    def test_extract_typescript_default_import(self):
        """Test extracting TypeScript default import."""
        source = "import React from 'react';"
        imports = extract_imports_from_source(source, "typescript", "/test.ts")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert "react" in imp.module

    def test_extract_typescript_relative_import(self):
        """Test extracting TypeScript relative import."""
        source = "import { helper } from './utils';"
        imports = extract_imports_from_source(source, "typescript", "/test.ts")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.is_relative is True

    def test_extract_go_import(self):
        """Test extracting Go import."""
        source = '''package main

import "fmt"
'''
        imports = extract_imports_from_source(source, "go", "/test.go")
        
        assert len(imports) >= 1
        imp = imports[0]
        assert imp.module == "fmt"

    def test_extract_go_grouped_imports(self):
        """Test extracting Go grouped imports."""
        source = '''package main

import (
    "fmt"
    "os"
)
'''
        imports = extract_imports_from_source(source, "go", "/test.go")
        
        # At least one import extracted
        assert len(imports) >= 1

    def test_unsupported_language(self):
        """Test with unsupported language."""
        imports = extract_imports_from_source("code", "unsupported", "/test.x")
        assert imports == []


class TestImportResolution:
    """Tests for import path resolution."""

    def test_extractor_with_project_root(self):
        """Test extractor with project root."""
        extractor = ImportExtractor(
            resolve_paths=False,
            project_root="/project",
        )
        assert extractor.project_root is not None


class TestConvenienceFunction:
    """Tests for convenience function."""

    def test_extract_imports_from_source(self):
        """Test the convenience function."""
        source = "import json"
        imports = extract_imports_from_source(source, "python")
        assert len(imports) >= 1

    def test_with_custom_path(self):
        """Test with custom file path."""
        source = "import os"
        imports = extract_imports_from_source(source, "python", "/custom/test.py")
        if imports:
            assert imports[0].file_path == "/custom/test.py"
