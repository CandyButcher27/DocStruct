# Makes `tests` a package rooted here.
# ultralytics ships a top-level `tests` package into site-packages; without this
# file `tests.conftest` resolves to theirs and the whole suite fails to collect.
