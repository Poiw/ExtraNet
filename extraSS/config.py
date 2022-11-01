Dataloader_Keys = [ "PreTonemapHDRColor" ,
                    "demodulatePreTonemapHDRColor",
                    "HighResoTAAPreTonemapHDRColor",
                    "Specular"          ,
                    "WorldNormal"       ,
                    "BaseColor"         ,
                    "Roughness"         ,
                    "Metallic"          ,
                    "SceneDepth"        ,
                    "occ-warp_demodulatePreTonemapHDRColor",
                    "warp_demodulatePreTonemapHDRColor",
                    "occ-warp_HighResoTAAPreTonemapHDRColor"
                    ]


mLossHoleArgument = 1
mLossHardArgument = 1

dataType = "SS-glossy_shading"

# train
SS_only_ratio = 0.01
# test
# SS_only_ratio = -1

low_weight = 3
high_weight = 0.5

loss_func_name = "Multireso_mLoss"

network_type = "ExtraNet_demodulate_noHistory_SS_blend"
input_channels = 13
output_channels = 3