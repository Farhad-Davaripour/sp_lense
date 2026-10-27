"""Narrow rendered Kaggle Logs status compatibility; no scientific admission.

The newer ``Successfully ran in ...s`` presentation is accepted only as the
page's AX status heading, bound to the primary browser and AX page identities.
An arbitrary log message or a success phrase alone is not this presentation.
"""
import math
import re
from urllib.parse import parse_qs, urlsplit


def _route(raw, notebook, run, *, ax=False):
    if ax and raw.startswith('kaggle.com/'):
        raw='https://www.'+raw
    value=urlsplit(raw)
    if (value.scheme!='https' or value.netloc!='www.kaggle.com'
            or value.path not in ['/code/'+notebook,'/code/'+notebook+'/log']
            or value.fragment or parse_qs(value.query,keep_blank_values=True)!={'scriptVersionId':[str(run)]}):
        return None
    return value.path,value.query


def rendered_terminal_success(text, notebook, run):
    """Recognize the observed rendered status, never text embedded in a table.

Legacy terminal strings remain handled by their existing callers. This new
format requires both primary page identities, the selected Logs tab, its Logs
heading, one exact level-five status heading and its matching child text.
"""
    if type(run) is not int or run<=0 or not isinstance(text,str):return False
    lines=text.splitlines()
    first=next((i for i,line in enumerate(lines) if line.strip()),None)
    if first is None:return False
    browser=re.fullmatch(r'Browser tab: \d+, Title: "[^\"]*", URL: "([^\"]+)"\.',lines[first])
    if browser is None:return False
    try:route=_route(browser[1],notebook,run)
    except ValueError:return False
    if route is None:return False
    # Rendered log rows cannot supply a terminal page-status heading.
    table=next((i for i,line in enumerate(lines) if re.match(r'^\s*\d+ table(?:\s|$)',line)),len(lines))
    prefix=lines[first+1:table]
    pages=[match for line in prefix if (match:=re.fullmatch(r'\s*\d+ AXWebArea .*, URL: (\S+)\s*',line))]
    if len(pages)!=1:return False
    try:
        if _route(pages[0][1],notebook,run,ax=True)!=route:return False
    except ValueError:return False
    selected=re.compile(r'^\s*\d+ tab \((?=[^)]*\bselected\b)[^)]*\) Description: Logs, \d+ of \d+, Value: 1\s*$')
    tabs=[i for i,line in enumerate(prefix) if selected.fullmatch(line)]
    logs=[(i,match) for i,line in enumerate(prefix)
          if (match:=re.fullmatch(r'(\s*)\d+ heading Logs, Value: 2\s*',line))]
    if len(tabs)!=1 or len(logs)!=1 or tabs[0]>=logs[0][0]:return False
    # A conflicting page-status heading fails closed, even if another positive
    # heading or quoted message is present in the same captured page.
    negative=re.compile(r'^\s*\d+ heading (?:Run )?(?:Failed|Failure|Cancelled|Canceled|Running|Error|Unsuccessful|Unsuccessfully|Not\b|Did not\b)',re.I)
    if any(negative.match(line) for line in prefix):return False
    success=re.compile(r'(\s*)\d+ heading (Successfully ran in ((?:0|[1-9]\d*)(?:\.\d+)?)s), Value: 5\s*')
    headings=[(i,match) for i,line in enumerate(prefix) if (match:=success.fullmatch(line))]
    if len(headings)!=1:return False
    index,heading=headings[0]
    if index<=logs[0][0] or len(heading[1].expandtabs(4))!=len(logs[0][1][1].expandtabs(4))+4:
        return False
    seconds=float(heading[3])
    if not math.isfinite(seconds) or seconds<0:return False
    child=next((line for line in prefix[index+1:] if line.strip()),'')
    match=re.fullmatch(r'(\s*)\d+ text '+re.escape(heading[2])+r'\s*',child)
    return match is not None and len(match[1].expandtabs(4))>len(heading[1].expandtabs(4))
