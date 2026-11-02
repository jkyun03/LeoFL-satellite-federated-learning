# LeoFL/followup/__init__.py
#
# Follow-up study: contact-aware local-step scheduling + FedNova for
# leader-based hierarchical satellite FL (LH-FL), with offloading disabled.
#
# This package is intentionally independent of LeoFL.Simulator's training/
# scheduling/aggregation code (scenarios.py, run_simulator.py). It reuses only
# shared assets (backbone/embeddings, data splitting, orbit + leader-selection
# utilities) from LeoFL.Simulator.fl_core / LeoFL.EuroSAT / LeoFL.Leo.
#
# See LeoFL/followup/README.md for the research design, assumptions, and
# how to run the included experiments.
