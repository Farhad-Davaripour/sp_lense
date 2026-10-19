"""Export the accepted Word manuscript to Markdown without discarding its equations or figures."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile
import xml.etree.ElementTree as ET
from src.config import ROOT

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
M = '{http://schemas.openxmlformats.org/officeDocument/2006/math}'
R = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'


def accepted_document(document):
    """Project the proposed final view without accepting revisions in the source file."""
    deleted_marks = {p for p in document.iter(W+'p')
                     if p.find(W+'pPr/'+W+'rPr/'+W+'del') is not None}
    deleted_rows = {r for r in document.iter(W+'tr') if r.find(W+'trPr/'+W+'del') is not None}
    deleted_cells = {c for c in document.iter(W+'tc') if c.find(W+'tcPr/'+W+'cellDel') is not None}
    def visit(parent):
        for child in list(parent):
            tag = child.tag.removeprefix(W)
            if child in deleted_rows or child in deleted_cells or child.tag in (W+'del', W+'moveFrom'):
                parent.remove(child)
            elif child.tag in (W+'ins', W+'moveTo'):
                visit(child)
                index = list(parent).index(child)
                for offset, value in enumerate(list(child)):
                    child.remove(value); parent.insert(index+offset, value)
                parent.remove(child)
            elif child.tag.startswith(W) and (tag.endswith('PrChange') or tag == 'tblGridChange'
                  or tag in ('moveFromRangeStart', 'moveFromRangeEnd', 'moveToRangeStart', 'moveToRangeEnd')):
                parent.remove(child)
            else:
                visit(child)
    visit(document)
    for parent in document.iter():
        # A deleted paragraph mark joins its surviving text to the next paragraph.
        for p in list(parent):
            if p not in deleted_marks: continue
            index = list(parent).index(p)
            following = list(parent)[index+1:index+2]
            content = [e for e in p if e.tag != W+'pPr']
            if following and following[0].tag == W+'p':
                nxt = following[0]; start = int(nxt.find(W+'pPr') is not None)
                for offset, e in enumerate(content):
                    p.remove(e); nxt.insert(start+offset, e)
                parent.remove(p)
            elif not content:
                parent.remove(p)
    for parent in document.iter():
        for child in list(parent):
            if child.tag == W+'tbl' and not child.findall(W+'tr'):
                parent.remove(child)
    return document


def math_text(node):
    tag = node.tag.removeprefix(M)
    if tag.endswith('Pr'):
        return ''
    if tag == 't':
        return node.text or ''
    child = lambda name: ''.join(math_text(e) for e in node.find(M+name))
    if tag in ('sSub', 'sSup', 'sSubSup'):
        base = '{'+child('e')+'}'
        return base+('_{'+child('sub')+'}' if tag != 'sSup' else '')+('^{'+child('sup')+'}' if tag != 'sSub' else '')
    if tag == 'f':
        return r'\frac{'+child('num')+'}{'+child('den')+'}'
    if tag == 'd':
        props = node.find(M+'dPr')
        def mark(name, default):
            e = props.find(M+name) if props is not None else None
            return e.get(M+'val', default) if e is not None else default
        return mark('begChr', '(')+child('e')+mark('endChr', ')')
    if tag in ('oMath', 'oMathPara', 'r', 'e', 'num', 'den', 'sub', 'sup'):
        return ''.join(math_text(e) for e in node)
    raise ValueError('Unsupported equation element: '+tag)


def export(source=ROOT/'paper/manuscript.docx', destination=ROOT/'paper/manuscript.md'):
    source = Path(source); destination = Path(destination)
    with zipfile.ZipFile(source) as z:
        doc = ET.fromstring(z.read('word/document.xml'))
        settings = ET.fromstring(z.read('word/settings.xml'))
        tracked = (any(e.tag in {W+t for t in ('ins', 'del', 'moveTo', 'moveFrom')} for e in doc.iter())
                   or settings.find(W+'trackRevisions') is not None)
        if tracked:
            doc = accepted_document(doc)
        rels = {e.get('Id'): e for e in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
        figures = {}; equations = []
        def inline(node):
            if node.tag in (M+'oMath', M+'oMathPara'):
                text = math_text(node); equations.append(text)
                return '$'+text+'$'
            if node.tag == W+'t':
                return node.text or ''
            if node.tag == W+'tab':
                return ' '
            if node.tag == W+'br':
                return '\n'
            if node.tag == A+'blip':
                rel = rels[node.get(R+'embed')]
                target = PurePosixPath(rel.get('Target'))
                if target.is_absolute() or '..' in target.parts or target.parts[0] != 'media' or rel.get('TargetMode') == 'External':
                    raise ValueError('Expected an embedded paper figure')
                name = 'figures/'+target.name
                figures[name] = z.read('word/'+str(target))
                return f'![Paper figure {len(figures)} see its following caption]({name})'
            if node.tag in (W+'pPr', W+'rPr', W+'del', W+'delText'):
                return ''
            return ''.join(inline(e) for e in node)
        def paragraph(node):
            style = node.find(W+'pPr/'+W+'pStyle')
            style = style.get(W+'val', '') if style is not None else ''
            text = inline(node).strip()
            prefix = {'Title': '# ', 'Heading1': '## ', 'Heading2': '### ',
                      'Heading3': '#### ', 'ListBullet': '- '}.get(style, '')
            return prefix+text if text else ''
        blocks = []
        for node in doc.find(W+'body'):
            if node.tag == W+'p':
                value = paragraph(node)
                if value: blocks.append(value)
            elif node.tag == W+'tbl':
                rows = [[' '.join(paragraph(p) for p in c.findall(W+'p')).replace('|', r'\|').replace('\n', ' ')
                         for c in row.findall(W+'tc')] for row in node.findall(W+'tr')]
                if rows:
                    blocks.append('\n'.join(['| '+' | '.join(rows[0])+' |',
                        '| '+' | '.join(['---']*len(rows[0]))+' |']+['| '+' | '.join(r)+' |' for r in rows[1:]]))
        destination.parent.mkdir(parents=True, exist_ok=True)
        for name, content in figures.items():
            path = destination.parent/name; path.parent.mkdir(exist_ok=True); path.write_bytes(content)
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        view = ' view of proposed revisions' if tracked else ''
        text = '<!-- Accepted manuscript'+view+'; source DOCX SHA-256 '+source_hash+' -->\n\n'+'\n\n'.join(blocks)+'\n'
        destination.write_text(text, encoding='utf-8', newline='\n')
        return {'source_docx_sha256': source_hash, 'markdown_bytes': destination.stat().st_size,
                'equations': len(equations), 'figures': len(figures),
                'tables': len(doc.findall('.//'+W+'tbl')), 'tracked_content_exported': False,
                'revision_view': 'proposed_final' if tracked else 'clean'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--source', type=Path, default=ROOT/'paper/manuscript.docx')
    p.add_argument('--output', type=Path, default=ROOT/'paper/manuscript.md')
    a = p.parse_args(); print(json.dumps(export(a.source, a.output)))
