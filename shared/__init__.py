"""Shared utilities used across case_generator, patient_agent, and doctor_agent."""

from .clients import CLIENT_CONFIGS, CLIENT_NAMES, DEFAULT_CLIENT, create_client

__all__ = ["CLIENT_CONFIGS", "CLIENT_NAMES", "DEFAULT_CLIENT", "create_client"]
