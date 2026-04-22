import json
import glob
import re

def get_indent(line):
    return len(line) - len(line.lstrip())

with open('full_functions_extracted.py', 'w', encoding='utf-8') as out:
    for f in glob.glob('notebooks/*.ipynb'):
        try:
            notebook = json.load(open(f, encoding='utf-8'))
        except:
            continue
        
        in_func = False
        func_indent = 0
        
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                lines = cell.get('source', [])
                if isinstance(lines, str):
                    lines = [lines]
                for line in lines:
                    line_str = line.replace('\r', '')
                    if line_str.startswith('def '):
                        if in_func:
                            out.write('\n')
                        out.write(f'# From {f}\n')
                        out.write(line_str)
                        if not line_str.endswith('\n'): out.write('\n')
                        in_func = True
                        func_indent = 0 # base indent
                    elif in_func:
                        if line_str.strip() == '':
                            out.write('\n')
                        elif get_indent(line_str) > 0 or line_str.startswith(')'):
                            out.write(line_str)
                            if not line_str.endswith('\n'): out.write('\n')
                        else:
                            in_func = False
                            out.write('\n')
