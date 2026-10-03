"""Flatten a LaTeX manuscript (macros, row files, citations, cross-references) and convert it to .docx with pandoc."""
import re, sys, subprocess, os
src, out, aux_main = sys.argv[1], sys.argv[2], sys.argv[3]
bbl = sys.argv[4] if len(sys.argv) > 4 else None
s = open(src).read()
# macros
macros = {}
for f in ['numbers.tex', 'supp_numbers.tex']:
    if os.path.exists(f):
        for m in re.finditer(r'\\newcommand\{\\(\w+)\}\{(.*)\}\s*$', open(f).read(), re.M):
            macros[m.group(1)] = m.group(2)
s = re.sub(r'\\input\{(numbers|supp_numbers)\.tex\}', '', s)
s = re.sub(r'\\makeatletter\\newcommand\{\\inputrows\}.*?\\makeatother', '', s)
s = re.sub(r'\\inputrows\{([\w.]+)\}', lambda m: open(m.group(1)).read(), s)
for _ in range(3):
    for k in sorted(macros, key=len, reverse=True):
        s = re.sub(r'\\' + k + r'(\{\})?(?![A-Za-z])', lambda m, k=k: macros[k], s)
# labels from aux
labels, cites = {}, {}
for aux in aux_main.split(','):
    for m in re.finditer(r'\\newlabel\{([^}]+)\}\{\{([^}]*)\}\{([^}]*)\}', open(aux).read()):
        labels[m.group(1)] = (m.group(2), m.group(3))
    for m in re.finditer(r'\\bibcite\{([^}]+)\}\{\{(\d+)\}', open(aux).read()):
        cites[m.group(1)] = int(m.group(2))
def cite(m):
    ns = sorted(cites[k.strip()] for k in m.group(1).split(','))
    parts, i = [], 0
    while i < len(ns):
        j = i
        while j + 1 < len(ns) and ns[j + 1] == ns[j] + 1: j += 1
        parts.append(f'{ns[i]}' if j == i else (f'{ns[i]},{ns[j]}' if j == i + 1 else f'{ns[i]}--{ns[j]}')); i = j + 1
    return '[' + ','.join(parts) + ']'
s = re.sub(r'~?\\cite\{([^}]+)\}', lambda m: ' ' + cite(m), s)
s = re.sub(r'\\pageref\{([^}]+)\}', lambda m: labels.get(m.group(1), ('', '?'))[1], s)
s = re.sub(r'\\ref\{([^}]+)\}', lambda m: labels.get(m.group(1), ('?', ''))[0], s)
def envfix(m):
    env, body = m.group(1), m.group(2)
    lab = re.search(r'\\label\{([^}]+)\}', body)
    num = labels.get(lab.group(1), ('', ''))[0] if lab else ''
    name = 'Figure' if env == 'figure' else 'Table'
    k = [0]
    def subcap(mm):
        k[0] += 1
        return mm.group(1) + '\\caption{(' + 'abcdefgh'[k[0] - 1] + ') '
    body = re.sub(r'(\\begin\{subfigure\}\{[^}]*\}[^\n]*?)\\caption\{', subcap, body)
    if num:
        idx = body.rfind('\\caption{') if env == 'figure' else body.find('\\caption{')
        if idx >= 0:
            body = body[:idx] + '\\caption{\\textbf{' + name + ' ' + num + '.} ' + body[idx + len('\\caption{'):]
    return '\\begin{' + env + '}' + body + '\\end{' + env + '}'
s = re.sub(r'\\begin\{(figure|table)\}(.*?)\\end\{\1\}', envfix, s, flags=re.S)
s = re.sub(r'\\label\{[^}]+\}', '', s)
# layout wrappers
s = s.replace('\\resizebox{\\textwidth}{!}{%\n', '')
s = re.sub(r'\\end\{(tabular|tabularx|tikzpicture)\}\}', r'\\end{\1}', s)
def makecell(m):
    return m.group(2).replace('\\\\', ' ')
for _ in range(2):
    s = re.sub(r'\\makecell(\[\w\])?\{((?:[^{}]|\{[^{}]*\})*)\}', makecell, s)
s = re.sub(r'\\multirow\{\d+\}\{\*\}\{((?:[^{}]|\{[^{}]*\})*)\}', r'\1', s)
s = re.sub(r'\\multicolumn\{(\d+)\}\{(?:[^{}]|\{[^{}]*\})*\}\{((?:[^{}]|\{[^{}]*\})*)\}', lambda m: m.group(2) + ' &' * (int(m.group(1)) - 1), s)
s = s.replace('\\cellcolor{green!12}', '').replace('\\cellcolor{gray!12}', '').replace('\\cellcolor{orange!15}', '')
s = s.replace('\\R ', 'Reported ').replace('\\NA ', 'N/A ').replace('\\PR ', 'Partial ').replace('\\NR ', 'Not in abstract ')
s = re.sub(r'\\newcolumntype\{[PY]\}.*\n', '', s); s = re.sub(r'\\renewcommand\{\\cellalign\}\{lc\}(\\renewcommand\{\\theadalign\}\{lc\})?', '', s)
s = re.sub(r'(\\begin\{(?:tabular|tabularx|longtable)\}(?:\{\\textwidth\})?)\{([^\n]*?)\}\n', lambda m: m.group(1) + '{' + m.group(2).replace('P{', 'p{').replace('Y', 'X') + '}\n', s)
s = re.sub(r'\\loc\{([^}]+)\}', lambda m: f"Sec. {labels.get(m.group(1), ('?', '?'))[0]}, p. {labels.get(m.group(1), ('?', '?'))[1]}", s)
s = s.replace('\\par\\vspace{3pt}{\\scriptsize\\raggedright\\setlength{\\parskip}{1pt}', '\n\n{\\small ').replace('\\par\\textsuperscript', '\n\n\\textsuperscript').replace('\\par ', '\n\n')
s = s.replace('\\tableofcontents', '')
s = re.sub(r'\\addcontentsline\{[^}]*\}\{[^}]*\}\{[^}]*\}', '', s)
s = re.sub(r'\\captionof\{figure\}\{', r'\\textbf{Figure.} {', s)
# bibliography
if bbl:
    b = open(bbl).read()
    items = re.split(r'\\bibitem\[[^\]]*\]\{[^}]+\}', b.replace('\n', ' '))[1:]
    order = re.findall(r'\\bibitem\[[^\]]*\]\{([^}]+)\}', b.replace('\n', ' '))
    lines = []
    for key, it in zip(order, items):
        it = it.replace('\\end{thebibliography}', '')
        it = re.sub(r'\\newblock\s*', ' ', it); it = it.replace('\\penalty0', '').replace('~', ' ')
        it = re.sub(r'\\doi\{([^}]+)\}', r'doi:\1', it)
        it = re.sub(r'\\url\{([^}]+)\}', r'\1', it)
        it = re.sub(r'\\natexlab\{([^}]*)\}', r'\1', it)
        it = re.sub(r'\s+', ' ', it).strip()
        lines.append(f'[{cites[key]}] {it}\n')
    s = s.replace('\\bibliographystyle{unsrtnat}', '').replace('\\bibliography{refs}', '\\section*{References}\n' + '\n'.join(lines))
open(out.replace('.docx', '_flat.tex'), 'w').write(s)
subprocess.run(['pandoc', out.replace('.docx', '_flat.tex'), '-f', 'latex', '-t', 'docx', '-o', out, '--resource-path=.', '--number-sections', '--reference-doc=ref.docx'], check=True)
print('ok', out)
