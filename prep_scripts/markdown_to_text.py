import os
import re

from settings import *


def split_path(path):
    components = []
    while True:
        path, tail = os.path.split(path)
        if tail == "":
            if path != "":
                components.append(path)
            break
        components.append(tail)
    components.reverse()
    return components


def remove_comments(md):
    return re.sub(r'<!--((.|\n)*)-->', '', md)


header_pattern = re.compile(r'\n\s*\n(#{1,3})\s(.*)\n\s*\n')


def split_content(content):
    text_chunk_size = context_lengths[EMBED_NAME] - 32
    _parts = content.split('\n\n')
    parts = []
    for p in _parts:
        if len(p) < text_chunk_size:
            parts.append(p)
        else:
            parts.extend(p.split('\n'))

    res = ['']
    for p in parts:
        if len(res[-1]) + len(p) < text_chunk_size:
            res[-1] += p + '\n\n'
        else:
            res.append(p + '\n\n')

    return res


def split_markdown(md):
    def construct_chunks(content):
        parts = split_content(content)
        for p in parts:
            construct_chunk(p)

    def construct_chunk(content):
        content = content.strip()
        if len(content) == 0:
            return

        chunk = ''
        for i in sorted(name_hierarchy):
            if len(name_hierarchy[i]) != 0:
                j = i + 1
                while j in name_hierarchy:
                    if name_hierarchy[j].find(name_hierarchy[i]) != -1:
                        break
                    j += 1
                else:
                    chunk += f'{"#" * (i + 1)}{name_hierarchy[i]}\n\n'

        chunk += content
        chunk = chunk.strip()
        res.append(chunk)

    # to find a header at the top of a file
    md = f'\n\n{md}'
    headers = list(header_pattern.finditer(md))
    # only first header can be first-level
    headers = [h for i, h in enumerate(headers) if i == 0 or len(h.group(1)) > 1]

    name_hierarchy = {i: '' for i in (1, 2, 3)}
    res = []
    for i in range(len(headers)):
        header = headers[i]
        level = len(header.group(1))
        name = header.group(2).strip()
        name_hierarchy[level] = name
        if i == 0 and header.start() != 0:
            construct_chunks(md[:header.start()])

        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else None
        construct_chunks(md[start:end])

    if len(headers) == 0:
        construct_chunks(md)

    return res

