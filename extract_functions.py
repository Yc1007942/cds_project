import json
import glob

with open('notebook_functions.txt', 'w', encoding='utf-8') as out:
    for f in glob.glob('notebooks/*.ipynb'):
        try:
            notebook = json.load(open(f, encoding='utf-8'))
        except:
            continue
        funcs = []
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                for line in cell.get('source', []):
                    if line.startswith('def '):
                        funcs.append(line.strip())
        if funcs:
            out.write(f'--- {f} ---\n')
            for func in funcs:
                out.write(f'{func}\n')
            out.write('\n')
