"""Bounded OOXML parsing with all DTD and entity declarations rejected."""

import xml.etree.ElementTree as ET
from xml.parsers import expat

MAX_XML_BYTES = 8 * 1024 * 1024


def parse_xml(data):
    """Use Expat events to build a tree without permitting entity expansion or I/O."""
    if len(data) > MAX_XML_BYTES:
        raise ValueError("XML part exceeds the size limit")
    builder = ET.TreeBuilder()
    parser = expat.ParserCreate(namespace_separator="}")
    depth, nodes = 0, 0

    def qualified(name):
        return "{" + name if "}" in name else name

    def reject_declaration(*args):
        raise ValueError("DTD and entity declarations are not allowed")

    def start(name, attributes):
        nonlocal depth, nodes
        depth += 1
        nodes += 1
        if depth > 128 or nodes > 200000:
            raise ValueError("XML structure exceeds the parsing limits")
        builder.start(qualified(name), {qualified(k): v for k, v in attributes.items()})

    def end(name):
        nonlocal depth
        builder.end(qualified(name))
        depth -= 1

    parser.StartElementHandler = start
    parser.EndElementHandler = end
    parser.CharacterDataHandler = builder.data
    parser.StartDoctypeDeclHandler = reject_declaration
    parser.EntityDeclHandler = reject_declaration
    parser.ExternalEntityRefHandler = reject_declaration
    parser.Parse(data, True)
    return builder.close()


def read_xml_part(archive, name):
    if archive.getinfo(name).file_size > MAX_XML_BYTES:
        raise ValueError("XML part exceeds the size limit")
    return archive.read(name)
