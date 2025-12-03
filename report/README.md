# LaTeX Report for ECE685 Project 2

This directory contains the LaTeX source for the NeurIPS-style final report.

## Files

- `main.tex` - Main LaTeX document
- `references.bib` - BibTeX references
- `neurips_2024.sty` - NeurIPS style file (you need to download this)
- `figures/` - Directory for figures (create and add your plots)

## Building the Report

### Option 1: Overleaf (Recommended)

1. Go to [Overleaf](https://www.overleaf.com/)
2. Create a new project
3. Upload `main.tex` and `references.bib`
4. Download the NeurIPS 2024 style from: https://neurips.cc/Conferences/2024/PaperInformation/StyleFiles
5. Upload `neurips_2024.sty` to the project
6. Create a `figures/` folder and upload your generated plots
7. Compile with pdfLaTeX

### Option 2: Local LaTeX

```bash
# Make sure you have texlive installed
# macOS: brew install --cask mactex
# Ubuntu: sudo apt-get install texlive-full

# Compile
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Required Figures

Generate these from your notebook and save to `figures/`:

1. `feature_correlations.png` - Bar chart of top features by correlation
2. `calibrated_steering_results.png` - Main results visualization

## Customization

### Author Information

Update line ~25 in `main.tex`:

```latex
\author{
  Your Name \\
  Department of Electrical and Computer Engineering\\
  Duke University\\
  Durham, NC 27708 \\
  \texttt{your.email@duke.edu} \\
}
```

### Results Tables

Update the numbers in Tables 2, 3, 4 with your actual experimental results.

### Abstract

Modify the abstract to highlight your specific findings.

## NeurIPS Style Notes

- Page limit is typically 8 pages (excluding references and appendix)
- Use `\citep{}` for parenthetical citations
- Use `\citet{}` for textual citations
- Figures should be high resolution (300 DPI minimum)
- Use vector graphics (PDF) when possible
