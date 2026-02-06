"""Performance benchmark: legacy (regex) vs unified (tree-sitter) pipeline.

Measures extraction and learning throughput for both pipelines to establish
a baseline before flipping the default `use_unified_pipeline` flag.

Run with:
    pytest tests/extraction_consolidation/test_pipeline_benchmark.py -v --benchmark-only
    pytest tests/extraction_consolidation/test_pipeline_benchmark.py -v --benchmark-columns=mean,stddev,rounds
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from anamnesis.services.learning_service import LearningOptions, LearningService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def benchmark_codebase(tmp_path: Path) -> Path:
    """Create a multi-file Python codebase of realistic size.

    Generates ~15 files with classes, functions, constants, imports,
    and design patterns (singleton, factory) to exercise both
    extraction paths meaningfully.
    """
    files: dict[str, str] = {}

    # service layer (3 files)
    files["services/user_service.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from dataclasses import dataclass
        from typing import Optional

        MAX_LOGIN_ATTEMPTS = 5
        SESSION_TIMEOUT_SECONDS = 3600

        @dataclass
        class UserCredentials:
            username: str
            password_hash: str

        class UserService:
            \"\"\"Manages user authentication and profiles.\"\"\"

            _instance: Optional[UserService] = None

            @classmethod
            def get_instance(cls) -> UserService:
                if cls._instance is None:
                    cls._instance = cls()
                return cls._instance

            def __init__(self):
                self._cache: dict[str, dict] = {}

            async def authenticate(self, creds: UserCredentials) -> bool:
                return True

            async def get_profile(self, user_id: str) -> dict:
                if user_id in self._cache:
                    return self._cache[user_id]
                return {"id": user_id}

            def invalidate_cache(self, user_id: str) -> None:
                self._cache.pop(user_id, None)
    """)

    files["services/order_service.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from typing import Optional
        from decimal import Decimal

        DEFAULT_CURRENCY = "USD"
        TAX_RATE = Decimal("0.08")

        class OrderItem:
            def __init__(self, product_id: str, quantity: int, price: Decimal):
                self.product_id = product_id
                self.quantity = quantity
                self.price = price

        class OrderService:
            \"\"\"Handles order creation and processing.\"\"\"

            def create_order(self, items: list[OrderItem]) -> dict:
                total = sum(i.price * i.quantity for i in items)
                tax = total * TAX_RATE
                return {"total": total, "tax": tax, "currency": DEFAULT_CURRENCY}

            def cancel_order(self, order_id: str) -> bool:
                return True

            def get_order_status(self, order_id: str) -> str:
                return "pending"
    """)

    files["services/notification_service.py"] = textwrap.dedent("""\
        from __future__ import annotations
        import logging
        from typing import Protocol

        logger = logging.getLogger(__name__)

        MAX_RETRY_COUNT = 3
        NOTIFICATION_BATCH_SIZE = 100

        class NotificationChannel(Protocol):
            def send(self, recipient: str, message: str) -> bool: ...

        class EmailChannel:
            def send(self, recipient: str, message: str) -> bool:
                logger.info("Sending email to %s", recipient)
                return True

        class SmsChannel:
            def send(self, recipient: str, message: str) -> bool:
                logger.info("Sending SMS to %s", recipient)
                return True

        class NotificationService:
            def __init__(self, channels: list[NotificationChannel]):
                self._channels = channels

            def notify(self, recipient: str, message: str) -> int:
                sent = 0
                for ch in self._channels:
                    if ch.send(recipient, message):
                        sent += 1
                return sent
    """)

    # models layer (3 files)
    files["models/user.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from dataclasses import dataclass, field
        from datetime import datetime
        from enum import Enum
        from typing import Optional

        class UserRole(Enum):
            ADMIN = "admin"
            USER = "user"
            GUEST = "guest"

        @dataclass
        class User:
            id: str
            name: str
            email: str
            role: UserRole = UserRole.USER
            created_at: datetime = field(default_factory=datetime.now)
            is_active: bool = True
    """)

    files["models/product.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from dataclasses import dataclass
        from decimal import Decimal
        from enum import Enum

        class ProductCategory(Enum):
            ELECTRONICS = "electronics"
            CLOTHING = "clothing"
            FOOD = "food"

        @dataclass
        class Product:
            id: str
            title: str
            price: Decimal
            category: ProductCategory
            stock: int = 0

        @dataclass
        class ProductReview:
            product_id: str
            rating: int
            comment: str
    """)

    files["models/order.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from dataclasses import dataclass, field
        from datetime import datetime
        from decimal import Decimal
        from enum import Enum
        from typing import Optional

        class OrderStatus(Enum):
            PENDING = "pending"
            CONFIRMED = "confirmed"
            SHIPPED = "shipped"
            DELIVERED = "delivered"
            CANCELLED = "cancelled"

        @dataclass
        class Order:
            id: str
            user_id: str
            items: list[dict] = field(default_factory=list)
            total: Decimal = Decimal("0")
            status: OrderStatus = OrderStatus.PENDING
            created_at: datetime = field(default_factory=datetime.now)

        @dataclass
        class ShippingInfo:
            address: str
            city: str
            country: str
            postal_code: str
            tracking_number: Optional[str] = None
    """)

    # utils layer (3 files)
    files["utils/validators.py"] = textwrap.dedent("""\
        from __future__ import annotations
        import re

        EMAIL_PATTERN = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$")
        MIN_PASSWORD_LENGTH = 8
        MAX_USERNAME_LENGTH = 50

        def validate_email(email: str) -> bool:
            return bool(EMAIL_PATTERN.match(email))

        def validate_password(password: str) -> bool:
            if len(password) < MIN_PASSWORD_LENGTH:
                return False
            has_upper = any(c.isupper() for c in password)
            has_digit = any(c.isdigit() for c in password)
            return has_upper and has_digit

        def validate_username(username: str) -> bool:
            return 1 <= len(username) <= MAX_USERNAME_LENGTH
    """)

    files["utils/formatters.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from decimal import Decimal

        CURRENCY_SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£"}

        def format_currency(amount: Decimal, currency: str = "USD") -> str:
            symbol = CURRENCY_SYMBOLS.get(currency, currency)
            return f"{symbol}{amount:.2f}"

        def format_percentage(value: float) -> str:
            return f"{value * 100:.1f}%"

        def truncate_text(text: str, max_length: int = 100) -> str:
            if len(text) <= max_length:
                return text
            return text[:max_length - 3] + "..."
    """)

    files["utils/cache.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from collections import OrderedDict
        from typing import Any, Optional

        DEFAULT_CACHE_SIZE = 256
        CACHE_TTL_SECONDS = 300

        class LRUCache:
            \"\"\"Simple LRU cache implementation.\"\"\"

            def __init__(self, max_size: int = DEFAULT_CACHE_SIZE):
                self._max_size = max_size
                self._cache: OrderedDict[str, Any] = OrderedDict()

            def get(self, key: str) -> Optional[Any]:
                if key in self._cache:
                    self._cache.move_to_end(key)
                    return self._cache[key]
                return None

            def put(self, key: str, value: Any) -> None:
                if key in self._cache:
                    self._cache.move_to_end(key)
                self._cache[key] = value
                if len(self._cache) > self._max_size:
                    self._cache.popitem(last=False)

            def clear(self) -> None:
                self._cache.clear()

            @property
            def size(self) -> int:
                return len(self._cache)
    """)

    # api layer (2 files)
    files["api/routes.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from typing import Any

        API_VERSION = "v2"
        BASE_PATH = f"/api/{API_VERSION}"

        class Router:
            def __init__(self):
                self._routes: dict[str, Any] = {}

            def register(self, path: str, handler: Any) -> None:
                self._routes[BASE_PATH + path] = handler

            def resolve(self, path: str) -> Any:
                return self._routes.get(path)
    """)

    files["api/middleware.py"] = textwrap.dedent("""\
        from __future__ import annotations
        import time
        import logging

        logger = logging.getLogger(__name__)
        REQUEST_TIMEOUT_MS = 30000

        class TimingMiddleware:
            def process_request(self, request: dict) -> dict:
                request["_start_time"] = time.monotonic()
                return request

            def process_response(self, request: dict, response: dict) -> dict:
                start = request.get("_start_time", 0)
                elapsed = (time.monotonic() - start) * 1000
                logger.debug("Request took %.1fms", elapsed)
                return response

        class AuthMiddleware:
            def __init__(self, secret_key: str):
                self._secret = secret_key

            def process_request(self, request: dict) -> dict:
                token = request.get("authorization", "")
                request["_authenticated"] = bool(token)
                return request
    """)

    # config (1 file)
    files["config/settings.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Optional

        DEFAULT_DB_HOST = "localhost"
        DEFAULT_DB_PORT = 5432
        LOG_LEVEL = "INFO"
        DEBUG_MODE = False

        @dataclass
        class DatabaseConfig:
            host: str = DEFAULT_DB_HOST
            port: int = DEFAULT_DB_PORT
            name: str = "app_db"
            user: str = "app"
            password: str = ""

        @dataclass
        class AppConfig:
            database: DatabaseConfig
            log_level: str = LOG_LEVEL
            debug: bool = DEBUG_MODE
            data_dir: Optional[Path] = None
    """)

    # factory pattern (1 file)
    files["factories.py"] = textwrap.dedent("""\
        from __future__ import annotations
        from typing import Any

        class ServiceFactory:
            \"\"\"Factory for creating service instances.\"\"\"

            _registry: dict[str, type] = {}

            @classmethod
            def register(cls, name: str, service_class: type) -> None:
                cls._registry[name] = service_class

            @classmethod
            def create(cls, name: str, **kwargs: Any) -> Any:
                if name not in cls._registry:
                    raise ValueError(f"Unknown service: {name}")
                return cls._registry[name](**kwargs)
    """)

    # main entry point (1 file)
    files["main.py"] = textwrap.dedent("""\
        from __future__ import annotations
        import sys

        APP_NAME = "benchmark_app"
        VERSION = "1.0.0"

        def main() -> int:
            print(f"{APP_NAME} v{VERSION}")
            return 0

        if __name__ == "__main__":
            sys.exit(main())
    """)

    # Write all files
    for rel_path, content in files.items():
        fp = tmp_path / rel_path
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)

    # Add __init__.py files
    for d in ["services", "models", "utils", "api", "config"]:
        (tmp_path / d / "__init__.py").write_text("")

    return tmp_path


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------


class TestPipelineBenchmark:
    """Benchmark legacy vs unified pipeline throughput."""

    def test_legacy_pipeline_throughput(self, benchmark, benchmark_codebase):
        """Benchmark the legacy (regex-based) pipeline."""
        def run_legacy():
            service = LearningService(use_unified_pipeline=False)
            return service.learn_from_codebase(
                benchmark_codebase, LearningOptions(force=True)
            )

        result = benchmark(run_legacy)
        assert result.success
        assert result.concepts_learned > 0

    def test_unified_pipeline_throughput(self, benchmark, benchmark_codebase):
        """Benchmark the unified (tree-sitter) pipeline."""
        def run_unified():
            service = LearningService(use_unified_pipeline=True)
            return service.learn_from_codebase(
                benchmark_codebase, LearningOptions(force=True)
            )

        result = benchmark(run_unified)
        assert result.success
        assert result.concepts_learned > 0

    def test_unified_finds_more_concepts(self, benchmark_codebase):
        """Non-benchmark: verify unified pipeline extracts richer data."""
        legacy = LearningService(use_unified_pipeline=False)
        unified = LearningService(use_unified_pipeline=True)

        lr = legacy.learn_from_codebase(benchmark_codebase, LearningOptions(force=True))
        ur = unified.learn_from_codebase(benchmark_codebase, LearningOptions(force=True))

        assert lr.success and ur.success
        # Unified should find at least as many concepts (tree-sitter + regex merge)
        assert ur.concepts_learned >= lr.concepts_learned
        # Report counts for manual inspection
        print(f"\n  Legacy concepts: {lr.concepts_learned}, patterns: {lr.patterns_learned}")
        print(f"  Unified concepts: {ur.concepts_learned}, patterns: {ur.patterns_learned}")


class TestExtractionBenchmark:
    """Benchmark raw extraction (without learning overhead)."""

    @pytest.fixture
    def large_python_source(self) -> str:
        """Generate a large Python source file for extraction benchmarking."""
        lines = ["from __future__ import annotations", "import os", "from typing import Optional", ""]
        for i in range(20):
            lines.append(f"MAX_VALUE_{i} = {i * 100}")
        lines.append("")

        for i in range(10):
            lines.extend([
                f"class Service{i}:",
                f'    """Service number {i}."""',
                "",
                "    def __init__(self):",
                f"        self._value_{i} = {i}",
                "",
                f"    async def process_item(self, item_id: str) -> dict:",
                f'        return {{"id": item_id, "value": self._value_{i}}}',
                "",
                f"    def get_status(self) -> str:",
                f'        return "active"',
                "",
            ])

        for i in range(15):
            lines.extend([
                f"def helper_function_{i}(x: int, y: int) -> int:",
                f'    """Helper function {i}."""',
                f"    return x + y + {i}",
                "",
            ])

        return "\n".join(lines)

    def test_tree_sitter_extraction(self, benchmark, large_python_source):
        """Benchmark tree-sitter extraction on a large file."""
        from anamnesis.extraction.backends.tree_sitter_backend import TreeSitterBackend

        backend = TreeSitterBackend()

        def extract():
            return backend.extract_all(large_python_source, "large.py", "python")

        result = benchmark(extract)
        assert len(result.symbols) > 0
        assert isinstance(result.patterns, list)

    def test_regex_extraction(self, benchmark, large_python_source):
        """Benchmark regex extraction on a large file."""
        from anamnesis.intelligence.semantic_engine import SemanticEngine

        engine = SemanticEngine()

        def extract():
            return engine.extract_concepts(large_python_source, "large.py", "python")

        result = benchmark(extract)
        assert len(result) > 0
