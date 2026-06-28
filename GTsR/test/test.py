from gtsr import GTsRunner

# runner = GTsRunner(checkpoint="free")
# result = runner.clean(
#     cif="ALIDIL_R.cif",
#     output="./free",
#     threshold=0.5,
# )
# print("free:", result)

# runner = GTsRunner(checkpoint="all")
# result = runner.clean(
#     cif="ALIDIL_R.cif",
#     output="./all",
#     threshold=0.5,
# )
# print("all:", result)


runner = GTsRunner(checkpoint="stability")
score = runner.stability(cif="./all/ALIDIL_R_gtsr.cif")
print("score:", score)
if score == 1:
    print("The cleaned structure is stable.")
else:
    print("The cleaned structure is not stable.")