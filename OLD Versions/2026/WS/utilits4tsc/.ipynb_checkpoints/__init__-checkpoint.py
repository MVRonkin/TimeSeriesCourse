# from jupyter_utilits import *
from .TSC_Nixtla_dump import *
from .rule_ad import *
from .plot_series import *
from .prophet_adapter import *

import os
import re
import nbformat
from typing import Optional
import matplotlib.pyplot as plt

def get_notebook_name():
    """Пытается получить имя текущего ноутбука любыми способами."""
    
    # Способ 1: JavaScript (работает в Jupyter Notebook/Lab в браузере)
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            # Получаем заголовок страницы через JS
            result = ipython.eval_js('document.title')
            if result and result != 'Jupyter Notebook':
                # "MyNotebook.ipynb - Jupyter Notebook" -> "MyNotebook.ipynb"
                name = result.split(' - ')[0].strip()
                if name.endswith('.ipynb'):
                    return name
    except:
        pass
    
    # Способ 2: ipynbname (если установлен и работает)
    try:
        import ipynbname
        return ipynbname.filename()
    except:
        pass
    
    # Способ 3: Последний изменённый .ipynb в папке
    try:
        cwd = os.getcwd()
        ipynb_files = [f for f in os.listdir(cwd) if f.endswith('.ipynb')]
        if ipynb_files:
            # Сортируем по времени модификации
            ipynb_files.sort(key=lambda f: os.path.getmtime(os.path.join(cwd, f)), reverse=True)
            return ipynb_files[0]
    except:
        pass
    
    return None

def plt_style_GOST(fig_size = (12, 2.0)):
    plt.rcParams.update({
        # ШРИФТ
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times"],
        "font.size": 11,                      # ГОСТ: 10–12 pt
    
        # ОСИ И ПОДПИСИ
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
    
        # РАЗМЕР ФИГУРЫ (A4, отчёты)
        # "figure.figsize": (6.5, 4.0),         # ~16.5 × 10 см
        "figure.figsize": fig_size,         
        "figure.dpi": 150,
        "savefig.dpi": 300,
    
        # СОХРАНЕНИЕ
        "savefig.format": "pdf",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    
        # ЛИНИИ И ОСИ
        "axes.linewidth": 1.0,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
    
        "lines.linewidth": 1.5,
        "patch.linewidth": 1.0,
    
        # СЕТКА
        "axes.grid": True,                   # ГОСТ: обычно без сетки
        "axes.axisbelow": True,
    
        # TEX
        "text.usetex": False,
    })
    
def generate_toc(notebook_path: Optional[str] = None, 
                 title: str = '## Содержание', 
                 max_lvl: int = 2,
                 auto: bool = True) -> str:
    """
    Генерирует оглавление для Jupyter Notebook.
    
    Args:
        notebook_path: Путь к файлу. Если None и auto=True — пытается определить сам.
        title: Заголовок оглавления.
        max_lvl: Максимальный уровень заголовков.
        auto: Если True — пытается найти имя автоматически.
    """
    # Автоматическое определение
    if notebook_path is None and auto:
        notebook_path = get_notebook_name()
    
    if notebook_path is None:
        return "**Ошибка:** Не удалось определить имя ноутбука. Запусти: `generate_toc('ИмяФайла.ipynb')`"

    if not os.path.exists(notebook_path):
        # Пробуем найти файл с таким именем в текущей папке
        cwd = os.getcwd()
        if os.path.basename(notebook_path) == notebook_path:
            # Имя без пути — ищем в текущей директории
            if os.path.exists(os.path.join(cwd, notebook_path)):
                notebook_path = os.path.join(cwd, notebook_path)
            else:
                return f"**Ошибка:** Файл `{notebook_path}` не найден в `{cwd}`\n\nДоступные файлы:\n" + '\n'.join([f"- `{f}`" for f in os.listdir(cwd) if f.endswith('.ipynb')][:5])
    
    # Читаем ноутбук
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
    except Exception as e:
        return f"**Ошибка:** Не удалось прочитать файл: {e}"

    # Парсим заголовки
    toc = [title]
    anchor_counter = {}
    
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
            
        for line in cell.source.split('\n'):
            match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
            if not match:
                continue
            
            level = len(match.group(1))
            if level > max_lvl:
                continue
                
            text = match.group(2).strip()
            
            # Генерация slug
            slug = text.lower()
            slug = re.sub(r'[^\w\s-]', '', slug)
            slug = re.sub(r'[\s_]+', '-', slug)
            slug = slug.strip('-')
            
            if slug in anchor_counter:
                anchor_counter[slug] += 1
                slug = f"{slug}-{anchor_counter[slug]}"
            else:
                anchor_counter[slug] = 0
            
            indent = '  ' * (level - 1)
            toc.append(f"{indent}* [{text}](#{slug})")
    
    return '\n'.join(toc)