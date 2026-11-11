# Source and audit

The experiment modules and tests are captured from the audited training source. The two report tools postprocess sealed receipts. The baseline patch adds full-vocabulary entropy, optional detached TIS weighting and retriever timing hooks. It uses zero lines of context and can be applied with `git apply --unidiff-zero` to the frozen 181505 baseline checkout. The benchmark data, original runtime, weight files and submission credentials are supplied in the private environment.
