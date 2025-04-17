import re


def replace_doc_links(text):
    def repl(match):
        doc_id = match.group(1)
        url = f"#{doc_id}"
        return f'<a href="{url}" onmouseover="document.getElementById(\'doc_{doc_id}\').style=\'border: 2px solid white;background:#f27618\'; display: block;" onmouseout="document.getElementById(\'doc_{doc_id}\').style=\'border: 1px solid white; background: none; display:none;\'" >[{doc_id}]</a>'
    
    rep = re.sub(r"\[doc ?(\d+)\]", repl, text)
    rep = re.sub(r"\[document ?(\d+)\]", repl, rep)
    rep = re.sub(r"\(doc ?(\d+)\)", repl, rep)
    rep = re.sub(r"\(document ?(\d+)\)", repl, rep)
    rep = re.sub(r"document ?(\d+)", repl, rep)
    rep = re.sub(r"document no. ?(\d+)", repl, rep)
    rep = re.sub(r"Document no. ?(\d+)", repl, rep)

    return rep


def reverse_doc_links(html):
    def repl(match):
        doc_id = match.group(1)
        return f"[doc {doc_id}]"

    # Match <a ...>[n]</a> where n is a number
    return re.sub(r'<a [^>]*href="#(\d+)"[^>]*>\[\d+\]</a>', repl, html)
