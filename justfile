evalio:
    touch evalio/pyproject.toml
    uv --verbose sync

loam:
    touch loam/pyproject.toml
    uv --verbose sync

stubs:
    pybind11-stubgen --numpy-array-use-type-var loam
    pybind11-stubgen --numpy-array-use-type-var evalio._cpp
    cp stubs/loam/* loam/python/loam/
    cp stubs/evalio/_cpp/* evalio/python/evalio/_cpp/
    rm -rf stubs