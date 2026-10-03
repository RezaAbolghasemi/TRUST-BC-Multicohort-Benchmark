import re, sys
def split_top(s, sep):
    out, depth, cur, i = [], 0, '', 0
    while i < len(s):
        if s.startswith(sep, i) and depth == 0:
            out.append(cur); cur = ''; i += len(sep); continue
        ch = s[i]
        if ch == '{': depth += 1
        elif ch == '}': depth -= 1
        cur += ch; i += 1
    out.append(cur); return out
def fix_spec(spec):
    out, i = '', 0
    while i < len(spec):
        ch = spec[i]
        if ch in 'p@>' and i + 1 < len(spec) and spec[i + 1] == '{':
            d, j = 0, i + 1
            while True:
                if spec[j] == '{': d += 1
                elif spec[j] == '}':
                    d -= 1
                    if d == 0: break
                j += 1
            tok = spec[i:j + 1]
            out += ('P' + tok[1:]) if ch == 'p' else tok; i = j + 1; continue
        out += {'c': 'l', 'r': 'l', 'X': 'Y'}.get(ch, ch); i += 1
    return out
def bold_header(h):
    rows = split_top(h, '\\\\')
    new = []
    for r in rows:
        if not r.strip(): new.append(r); continue
        cells = split_top(r, '&')
        cells = [c if (not c.strip() or c.strip().startswith('\\textbf')) else re.sub(r'^(\s*)(.*?)(\s*)$', lambda m: m.group(1) + '\\textbf{' + m.group(2) + '}' + m.group(3), c, flags=re.S) for c in cells]
        new.append('&'.join(cells))
    return '\\\\'.join(new)
def process(s):
    def env(m):
        name, pre, spec, body = m.group(1), m.group(2) or '', m.group(3), m.group(4)
        t = body.find('\\toprule'); md = body.find('\\midrule')
        if t >= 0 and md > t:
            body = body[:t + 8] + bold_header(body[t + 8:md]) + body[md:]
        return f'\\begin{{{name}}}{pre}{{{fix_spec(spec)}}}{body}\\end{{{name}}}'
    pat = r'\\begin\{(tabular|tabularx|longtable)\}(\{\\textwidth\})?\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}(.*?)\\end\{\1\}'
    return re.sub(pat, env, s, flags=re.S)
for f in sys.argv[1:]:
    s = open(f).read()
    s = process(s)
    if '\\newcolumntype{P}' not in s:
        s = s.replace('\\begin{document}', '\\newcolumntype{P}[1]{>{\\raggedright\\arraybackslash}p{#1}}\n\\newcolumntype{Y}{>{\\raggedright\\arraybackslash}X}\n\\begin{document}', 1)
    open(f, 'w').write(s)
