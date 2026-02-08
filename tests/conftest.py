"""
Pytest configuration and shared fixtures for Anamnesis tests.
"""
import os

# Tests use temporary directories outside cwd — allow unrestricted roots in test mode.
os.environ.setdefault("ANAMNESIS_ALLOWED_ROOTS", "*")
