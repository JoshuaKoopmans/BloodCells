import os
if os.environ.get("PREFIX") is None:
    PREFIX = "/mnt/cellstorage/"
else:
    PREFIX = os.environ.get("PREFIX")