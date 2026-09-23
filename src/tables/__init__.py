"""Table structure recovery, chunking and region cropping.

Independent of the frozen src/vision/table_structure.py, which stays untouched
in the vision freeze scope. This package has its own freeze group
(scripts/table_freeze.py) because its inputs include images and fixtures that
no import analysis can reach.
"""
