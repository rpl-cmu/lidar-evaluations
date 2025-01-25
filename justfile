evalio:
    touch evalio/pyproject.toml
    uv --verbose sync
    pybind11-stubgen --numpy-array-wrap-with-annotated evalio._cpp -o evalio/python --ignore-all-errors

loam:
    touch loam/pyproject.toml
    uv --verbose sync
    pybind11-stubgen --numpy-array-wrap-with-annotated loam -o loam/python/ --ignore-all-errors

stubs:
    pybind11-stubgen --numpy-array-wrap-with-annotated evalio._cpp -o evalio/python
    pybind11-stubgen --numpy-array-wrap-with-annotated loam -o loam/python/

compdb:
    compdb -p loam/build/cp311-cp311-linux_x86_64/ -p evalio/build/cp311-cp311-linux_x86_64/ list > compile_commands.json

args:
    eval "$(register-python-argcomplete evalio)"