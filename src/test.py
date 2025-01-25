from serde.json import from_json, to_json
from serde.toml import from_toml, to_toml
from serde.yaml import from_yaml, to_yaml

import params

experiment_params = params.ExperimentParams(
    name="test",
    dataset="newer_college_2020/01_short_experiment",
    features=[params.Feature.Planar],
)

# print("JSON format:")
# print(to_json(experiment_params))

print("\nYAML format:")
out = to_yaml(
    experiment_params,
)
out = "# " + out.replace("\n", "\n# ")
print(out)

again = out.replace("# ", "")
print(again)

# print("\nTOML format:")
# print(to_toml(experiment_params))
