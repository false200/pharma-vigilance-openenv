"""Compatibility wrapper exposing the main environment class under server/."""

try:
    from ..env import PharmaVigilanceEnv
except ImportError:
    from env import PharmaVigilanceEnv

__all__ = ["PharmaVigilanceEnv"]
