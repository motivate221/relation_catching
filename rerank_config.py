TOPK_RETRIEVAL = 20
TOPM_RERANK = 3
EVIDENCE_SENT_NUM = 3

# Routing configuration.
# In the first integration stage we still send reranked candidates
# to multiple-choice, but we keep these thresholds for later extension.
DIRECT_VERIFY_THRESHOLD = 0.72
DIRECT_VERIFY_GAP = 0.10

# Score weights for relation reranking.
RETRIEVAL_WEIGHT = 0.40
SUMMARY_WEIGHT = 0.25
EVIDENCE_WEIGHT = 0.25
TYPE_WEIGHT = 0.05
DISTANCE_WEIGHT = 0.05

USE_TYPE_CONSTRAINT = True
