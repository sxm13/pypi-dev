<h1 align="center">GTsR</h1>

GTsR is a **g**raph neural network based **t**ool for solvent identification, **s**olvent **r**emoval, and activation-stability prediction in metal-organic frameworks (MOFs).


<div align="center">
        <img src="https://raw.githubusercontent.com/Xiao-Yan-Li-group/Web-Interface/main/imgs/gtsr_logo.png" alt="GTsR logo" width="500"/>
</div> 

                      
                       
![GitHub repo size](https://img.shields.io/github/repo-size/Xiao-Yan-Li-Group/GTsR?logo=github&logoColor=white&label=Repo%20Size)[![PyPI](https://img.shields.io/pypi/v/gtsr?logo=pypi&logoColor=white)](https://pypi.org/project/gtsr?logo=pypi&logoColor=white)[![Requires Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white)](https://python.org/downloads)[![GitHub license](https://img.shields.io/github/license/Xiao-Yan-Li-Group/GTsR.svg)](https://github.com/Xiao-Yan-Li-Group/GTsR/blob/main/LICENSE)

### Pretrained Models

| Checkpoint | File | Purpose |
| --- | --- | --- |
| `free` | `ckpt/free_best.pth` | Remove free solvent |
| `all` | `ckpt/all_best.pth` | Remove all solvent |
| `stability` | `ckpt/stability_best.pkl` | Predict activation stability |

## Installation

```bash
git clone https://github.com/coollkr/GTsR.git
cd GTsR
conda env create -f environment.yml
conda activate gtsr
pip install -e .
```

or

```bash
conda install -c conda-forge zeopp-lsmo
pip install gtsr
```

### Usage

- Remove solvent

```python
from gtsr import GTsRunner

runner = GTsRunner(checkpoint="free")
result = runner.clean(
    cif="input.cif",
    output="prediction",
    threshold=0.5,
)
```

You can also use:

```python
runner = GTsRunner(checkpoint="all")
runner = GTsRunner(checkpoint="path/to/ckpt.pth", device="cpu")
```

`runner.clean()` returns a dictionary with the following fields:

| Field | Description |
| --- | --- |
| `input` | Absolute path to the input CIF |
| `output` | Output directory |
| `framework` | Path to the cleaned framework CIF |
| `solvent` | Path to the solvent CIF, or `None` if not generated |
| `checkpoint` | Path to the checkpoint used for prediction |
| `task` | Task name stored in the checkpoint |
| `threshold` | Atom classification threshold |
| `num_atoms` | Total number of atoms |
| `num_framework_atoms` | Number of framework atoms |
| `num_solvent_atoms` | Number of solvent atoms |
| `probabilities` | Solvent probability for each atom |
| `labels` | Predicted class label for each atom |
| `solvent_smiles` | SMILES strings of identified solvents |

- Predict activation stability

```python
from gtsr import GTsRunner

runner = GTsRunner(checkpoint="stability")
score = runner.stability(cif="cleaned_framework.cif")

if score == 1:
    print("The cleaned structure is stable.")
else:
    print("The cleaned structure is not stable.")
```

## Web Interface

[Streamlit demo](https://xiao-yan-li-group-nus.streamlit.app/GTsR)

## Citation

Update the following entry when the associated publication becomes available:

```bibtex
@article{gtsr-xyl-group,
  title   = {GTSR: A GNN Based Tool for Solvent Removal from MOF with Stability Check},
  author  = {Liang, Kairui and Zhao, Guobin and Li, Xiao-Yan},
  year    = {2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
