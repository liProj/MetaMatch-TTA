"""Compile the English main paper and supplementary material with XeLaTeX."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
for iteration in range(3):
    for name in ['research_manuscript_en', 'research_supplement_en']:
        subprocess.run(['xelatex', '-interaction=nonstopmode', '-halt-on-error', name + '.tex'], cwd=root, check=True)
print('English main paper and supplementary material compiled.')
