<h1 align="center">ToBaCCo MOF</h1>

<h4 align="center">

</h4>              


[![Requires Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg?logo=python&logoColor=white)](https://python.org/downloads) [![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/sxm13/pypi-dev/LICENSE)![Build Status](https://img.shields.io/badge/build-passing-brightgreen) [![Gmail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:sxmzhaogb@gmail.com) [![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)]() [![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)]() 

#### useage
```sh    
from tobacco import run_tobacco

node = "./database/example/nodes/examplecif"
edge = "./database/example/edges/examplecif"
template = "./database/example/templates/example.cif"

run_tobacco(template, [node], [edge], save_path)

```


*   node: CIF file of metal node
*   edge: CIF file of linker
*   template:  CIF file topology                        
*   save_path: the path for construction of MOF     
                           

#### Reference
[Topologically Guided, Automated Construction of Metal–Organic Frameworks and Their Evaluation for Energy-Related Applications](https://doi.org/10.1021/acs.cgd.7b00848), 2017, 17, 11, 5801–5810.    
                                            

#### Bugs
If you encounter any problem, please email ```sxmzhaogb@gmail.com```.      