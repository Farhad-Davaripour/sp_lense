"""Conservative prose boundaries; XML calls and quoted arguments stay atomic."""
import re

ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'vs', 'etc', 'e.g', 'i.e', 'st', 'fig', 'no'}


def completed_boundaries(text, final=False):
    events = []
    quote = None
    escaped = False
    tool = False
    index = 0
    while index < len(text):
        if tool:
            if text.startswith('</tool_call>', index):
                index += len('</tool_call>')
                events.append({'kind': 'tool_call_complete', 'offset': index})
                tool = False
            else:
                index += 1
            continue
        if text.startswith('<tool_call>', index):
            tool = True
            index += len('<tool_call>')
            continue
        char = text[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if quote:
            if char == '\\':
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char in ('"', "'"):
            contraction = char == "'" and index > 0 and index + 1 < len(text) and text[index - 1].isalnum() and text[index + 1].isalnum()
            if not contraction:
                quote = char
            index += 1
            continue
        if char in '.!?':
            end = index + 1
            while end < len(text) and text[end] in '.!?':
                end += 1
            next_known = end < len(text) or final
            separated = end == len(text) or text[end].isspace()
            decimal = char == '.' and index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit()
            prefix = text[:index]
            match = re.search(r'([A-Za-z](?:[A-Za-z.]*)?)$', prefix)
            word = match.group(1).lower() if match else ''
            acronym = bool(re.fullmatch(r'(?:[A-Za-z]\.)+[A-Za-z]', word))
            abbreviation = char == '.' and (word in ABBREVIATIONS or acronym or (len(word) == 1 and word.isalpha()))
            if next_known and separated and not decimal and not abbreviation:
                events.append({'kind': 'sentence_complete', 'offset': end})
            index = end
            continue
        index += 1
    return events


class BoundaryTracker:
    def __init__(self):
        self.delivered = set()

    def update(self, decoded_text, final=False):
        fresh = []
        for event in completed_boundaries(decoded_text, final):
            identity = (event['kind'], event['offset'])
            if identity not in self.delivered:
                self.delivered.add(identity)
                fresh.append(event)
        return fresh
