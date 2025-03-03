evalio:
    touch src/evalio/pyproject.toml
    uv --verbose sync
    pybind11-stubgen --numpy-array-wrap-with-annotated evalio._cpp -o src/evalio/python --ignore-all-errors

loam:
    touch src/loam/pyproject.toml
    uv --verbose sync
    pybind11-stubgen --numpy-array-wrap-with-annotated loam -o src/loam/python/ --ignore-all-errors

stubs:
    pybind11-stubgen --numpy-array-wrap-with-annotated evalio._cpp -o src/evalio/python
    pybind11-stubgen --numpy-array-wrap-with-annotated loam -o src/loam/python/

compdb:
    compdb -p src/loam/build/cp311-cp311-linux_x86_64/ -p src/evalio/build/cp311-cp311-linux_x86_64/ list > compile_commands.json