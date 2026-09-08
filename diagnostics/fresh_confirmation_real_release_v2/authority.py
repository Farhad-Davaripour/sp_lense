"""Shared live/saved identity after exclusive trusted-root admission."""
import os
from real_boundary import Boundary,need


def current():
    return Boundary(),os.environ.get("SP_CONFIRMATION_AUTHORITY_SHA",""),os.environ.get("SP_CONFIRMATION_ADMISSION_SHA","")


def authenticate():
    boundary,approved,admission=current()
    return boundary.reauthenticate(approved,admission)


def independent_execution(saved):
    expected=authenticate()["execution"]
    need(saved==expected,"D_SAVED_EXECUTION_IDENTITY")
    return expected
