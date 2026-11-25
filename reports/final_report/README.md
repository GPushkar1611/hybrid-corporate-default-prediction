# AI in Finance course project report template

This editable template is a **writing guide** for course projects, not a
completed project report. The PDF shows the suggested structure. Replace all
guidance, bracketed placeholders, example questions, and sample tables with
your own evidence before submission. Do not invent observations or results.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Cover, abstract, document style, table of contents, references section |
| `body.tex` | Expanded guidance for every section of the report |
| `references.bib` | Space for actual BibTeX citations |
| `assets/iith-logo.png` | Institute logo used on the cover |
| `Makefile` | One-command PDF build |

A PDF preview is supplied separately. It is an output, so it is not needed in
the source archive.

## Write your report

1. Edit the project title, authors, course term, date, and abstract in `main.tex`.
2. Replace the guidance in `body.tex` with your completed project: introduction
   and related work; problem statement; data and methods; observed results;
   and conclusions. Delete irrelevant examples or subsections.
3. Add complete entries to `references.bib`. Cite them as `\cite{yourKey}`
   in `body.tex`. In `main.tex`, delete the References guidance and uncomment
   `\bibliographystyle{plain}` and `\bibliography{references}`.
4. Put any project-specific figures in `assets/` and include them with
   `\includegraphics{assets/figure-name.pdf}`. Cite data and borrowed visuals.
5. Compile with `make` (requires `latexmk` and a standard pdfLaTeX installation).
   Or run `pdflatex main.tex` twice if you do not have `latexmk`. If you add
   BibTeX citations without `latexmk`, run `pdflatex main.tex`,
   `bibtex main`, `pdflatex main.tex`, and `pdflatex main.tex`.

When experiments are incomplete, report what was actually done and state what
remains. Keep information availability, evaluation dates, baseline comparisons,
and limitations explicit for financial applications.

## Optional methodology reading

- [Time-aware cross-validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html)
  and [common data-leakage pitfalls](https://scikit-learn.org/stable/common_pitfalls)
  from the scikit-learn documentation.
- [AI risks and governance in finance](https://www.bis.org/publications/fsi-insight-63-regulating-ai-financial-sector-recent-developments-and-main-challenges)
  from the Bank for International Settlements.
