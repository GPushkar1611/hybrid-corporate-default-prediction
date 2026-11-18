# QUARANTINED: invalid SEC FSDS NUM panel

**Do not use this CSV for modeling or evaluation.** SEC Financial Statement Data Sets NUM values do not include the XBRL `scale` attribute. Treating them as dollar values produced impossible accounting ratios in this panel.

This artifact is retained only as a reproducible diagnostic record of the failed first construction. Its SEC SUB filing metadata can be used; its NUM financial values cannot. The active dataset is [`../sec_companyfacts_bankruptcy_v1/README.md`](../sec_companyfacts_bankruptcy_v1/README.md), which uses normalized SEC CompanyFacts values matched to exact filing accessions and reporting-period dates.

See the SEC [Financial Statement Data documentation](https://www.sec.gov/files/financial-statement-data-sets.pdf) for the NUM scale limitation.
