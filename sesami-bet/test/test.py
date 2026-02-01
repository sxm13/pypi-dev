from SESAMI.bet import fitbet
from SESAMI.predict import betml

BET_dict, BET_ESW_dict = fitbet(csv_file="ac_low.csv", columns=["P","N"],
                   adsorbate="N2", p0=1e5, T=77,
                   R2_cutoff=0.9995, R2_min=0.998,
                   font_size=12, font_type="DejaVu Sans",
                   legend=True, dpi=600, save_fig=True, verbose=False)
print(BET_dict, BET_ESW_dict)

MLBET = betml(csv_file="ac_low.csv", columns=["P","N"], verbose=False)
print(MLBET)