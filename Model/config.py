import torch
import os
mSkip=True
mLossHoleArgument=1
mLossHardArgument=1
root_dir="/home/poi/Data/Research/extraNet/Data"
basePaths = [os.path.join(root_dir, "RedwoodForest_Compressed_New", "RedwoodForest_updated")]  ## Path to all compressed data set folders


TestMetalicPrefix = "Metallic"

NormalPrefix="WorldNormal"
DepthPrefix="SceneDepth"
RoughnessPrefix="Roughness"

PreTonemapHDRColor="PreTonemapHDRColor"

warpPrefix = "Warp"

ofwarpPrefix = "WarpDiffuse"


TestNormalPrefix="WorldNormal"
TestDepthPrefix="SceneDepth"
TestRoughnessPrefix="Roughness"
mdevice=torch.device("cuda:0")

#Training related
learningrate=1e-3
epoch=100
printevery=50
batch_size=2
