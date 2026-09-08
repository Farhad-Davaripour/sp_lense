"""Explicit immutable production_v2 source provider; installing source is not invoking it."""
from pathlib import Path

COMMIT="f31d0e4f8fd5136eb38df01201f7032c9b82dc61"
PREFIX="diagnostics/fresh_confirmation_production_v2/"


def install(namespace,name):
    from support import SOURCES
    raw=SOURCES.read(COMMIT,PREFIX+name+".py")
    exec(compile(raw,namespace["__file__"],"exec"),namespace)


def module(name):
    from support import SOURCES
    return SOURCES.load("real_release_pinned_"+name,COMMIT,PREFIX+name+".py")
