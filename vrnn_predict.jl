using Revise

includet("vrnn_helper.jl")

dataset = makedata()
modf, args = load_mod("vaemods/model_back=false.bson")
modb, args = load_mod("vaemods/model_back=true.bson")



dfmod, dfnaive = run_predictions(dataset[1], modf, modb);
dfmod ./ dfnaive

