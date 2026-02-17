from tobacco import run_tobacco
from glob import glob
from tqdm import tqdm

save_path = "./test"

# nodes_dataset = glob("./database/example/nodes/*cif")
# edges_dataset = glob("./database/example/edges/*cif")
# templates_dataset = glob("./database/example/templates/*cif")

# for node in tqdm(nodes_dataset):
#     print("NODE:", node)
#     for edge in (edges_dataset):
#         print("EDGE:", edge)
#         for template in (templates_dataset):
#             try:
#                 run_tobacco(template, [node], [edge], save_path)
#             except:
#                 pass

run_tobacco("./test/acs.cif",
            ["./test/MIL-88B-dry_ASR_sbu_0.cif"], 
            ["./test/MIL-88B-dry_ASR_linker_0.cif"],
            save_path)