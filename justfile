evalio:
    touch evalio/pyproject.toml
    uv --verbose sync
    pybind11-stubgen --numpy-array-use-type-var evalio._cpp -o evalio/python --ignore-all-errors

loam:
    touch loam/pyproject.toml
    uv --verbose sync
    pybind11-stubgen --numpy-array-use-type-var loam -o loam/python/ --ignore-all-errors

compdb:
    compdb -p loam/build/cp311-cp311-linux_x86_64/ -p evalio/build/cp311-cp311-linux_x86_64/ list > compile_commands.json

args:
    eval "$(register-python-argcomplete evalio)"