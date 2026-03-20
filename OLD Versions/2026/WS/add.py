from IPython.display import Markdown, display
import re
import nbformat
import ipynbname
import pprint

def generate_toc(notebook_path='.', title='## Содержание'):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    toc = [title]
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            matches = re.findall(r'^(#+)\s+(.+)$', cell.source, flags=re.MULTILINE)
            for header in matches:
                level = len(header[0])
                text = header[1]
                slug = text.lower().replace(' ', '-')
                toc.append(f"{'  ' * (level - 1)}* [{text}](#{slug})")
    return '\n'.join(toc)
# Вывод оглавления в текущий ноутбук
try:
    name_ipynb = ipynbname.name()
    print(name_ipynb)   
except Exception as e:
    print("Не удалось получить имя")
display(Markdown(generate_toc(name_ipynb+'.ipynb')))