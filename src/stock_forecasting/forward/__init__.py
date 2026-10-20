"""Independent forward-inference pipeline for frozen signal checkpoints.

Forward data never enters a training dataset. A forward input package is built from
freshly fetched bars with the unchanged training data functions, so model inputs match
those of the validation/test exports without touching training splits, readiness or the
data-pipeline fingerprint (experiment protocol section 13.8 and its supplement).
"""
