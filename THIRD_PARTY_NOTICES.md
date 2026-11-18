# Third-party code

CAF-Mamba source and the Mamba reference operations are provided with their
license notices in the vendor directory.

The SEGA, DepMamba, MTSP, and PTTSD drivers use author modules staged by
prepare_baselines.py. The upstream/sources.json file records their repositories,
commits, and checksums. The patches under upstream adapt imports, tensor
devices, and modality inputs for the evaluation protocols.
