"""Convert the article-class master ../main.tex into the Robotica (ROB-New.cls) version ../main_rob.tex.
   python scripts/make_rob_version.py [--last-page N]
The master stays in the standard article class (all patch and check scripts work on it); the ROB version is generated from it, so edit main.tex, never main_rob.tex.
Needs ROB-New.cls and roblike.bst in the folder of main_rob.tex (they come from the official author package ROB-AuthorMacro.zip of Cambridge Core;
copies live in ../robotica_template/).  What is changed:
  * document class and packages (the class loads hyperref, booktabs, natbib [numbers], amsmath, graphicx, xcolor, ... itself);
  * front matter in the ROB format (authormark, article type, running pages, authors, affiliations, keywords, abstract);
  * the unnumbered declaration sections become the `con` block with \\ctitle headings (Robotica: Author Contributions, Financial Support, Conflicts of Interest,
    Ethical Approval, Data Availability, Acknowledgements incl. the generative-AI disclosure);
  * bibliography style roblike."""
import argparse, re, sys

ap = argparse.ArgumentParser()
ap.add_argument("--last-page", type=int, default=40)
ap.add_argument("--src", default="../main.tex")
ap.add_argument("--dst", default="../main_rob.tex")
a = ap.parse_args()

t = open(a.src, encoding="utf-8").read()

# ---- 1. document class and packages
i = t.index(r"\documentclass[11pt,a4paper]{article}")
j = t.index(r"\newcommand{\bench}")
t = t[:i] + "\\documentclass[DTMColor]{ROB-New}\n\\usepackage{bm,array,tabularx,enumitem}\n\n" + t[j:]

# ---- 2. front matter
m = re.search(r"\\title\{(.*?)\}\n\\author\{.*?\}\n\\date\{\}\n\n\\begin\{document\}\n\\maketitle\n\n\\begin\{abstract\}\n(.*?)\n\\end\{abstract\}\n", t, flags=re.S)
assert m, "front matter not found"
title, abstract = m.group(1).replace("\\\\\n", " ").replace("\\\\", " "), m.group(2)      # no manual line break in the running title
front = (
    "\\begin{document}\n\n"
    "\\authormark{First Author \\textit{et al.}}\n"
    "\\articletype{RESEARCH ARTICLE}\n"
    f"\\jnlPage{{1}}{{{a.last_page}}}\n"
    "\\jyear{2026}\n\n"
    f"\\title{{{title}}}\n\n"
    "\\author[1]{First Author}\n"
    "\\author[1]{Second Author}\n"
    "\\author[1]{Corresponding Author\\hyperlink{corr}{*}}\n"
    "\\address[1]{Department / Laboratory, University, City, Country}\n"
    "\\address{\\hypertarget{corr}{*}Corresponding author. \\email{email" "@" "example.com}}\n\n"
    "\\keywords{robotic harvesting, imitation learning, language grounding, counterfactual evaluation, simulation benchmark}\n\n"
    "\\abstract{" + abstract.strip() + "}\n\n"
    "\\maketitle\n"
)
t = t[:m.start()] + front + t[m.end():]

# ---- 3. declarations -> con block
i = t.index("\\section*{Author Contributions}")
j = t.index("\\clearpage\n\\bibliographystyle{unsrt}")
block = t[i:j]
parts = re.split(r"\\section\*\{([^}]*)\}\n", block)
# parts = ['', title1, body1, title2, body2, ...]
con = "\\clearpage\n\\begin{con}\n"                                   # flush the appendix floats before the back matter
for k in range(1, len(parts), 2):
    con += f"\\ctitle{{{parts[k]}}}\n{parts[k + 1].strip()}\n\n"
con += "\\end{con}\n\n"
t = t[:i] + con + t[j:]
t = t.replace("\\clearpage\n\\bibliographystyle{unsrt}", "\\bibliographystyle{roblike}")


# ---- 4. the class adds a full stop after every caption and run-in paragraph heading, so ours are stripped
def strip_final_period(text, macro):
    out, pos = [], 0
    while True:
        i = text.find(macro + "{", pos)
        if i < 0:
            out.append(text[pos:])
            break
        j, depth = i + len(macro) + 1, 1
        while depth:                                                         # matching brace of the argument
            c = text[j]
            if c == "\\":
                j += 2
                continue
            depth += (c == "{") - (c == "}")
            j += 1
        arg = text[i + len(macro) + 1: j - 1]
        stripped = arg.rstrip()
        if stripped.endswith(".") and not stripped.endswith("..") and not stripped.endswith("\\."):
            arg = stripped[:-1]
        out.append(text[pos:i + len(macro) + 1] + arg + "}")
        pos = j
    return "".join(out)


t = strip_final_period(t, "\\caption")
t = strip_final_period(t, "\\paragraph")

open(a.dst, "w", encoding="utf-8", newline="\n").write(t)
print("wrote", a.dst, len(t), "chars")
