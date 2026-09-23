"""Crop -> vision review -> OCR/VLM fusion -> re-decision.

Default mode is `disabled`. Nothing in this package calls a paid API unless
the config says live AND the caller passes allow_live=True.
"""
