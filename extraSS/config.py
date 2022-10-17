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
                    "occ-warp_HighResoTAAPreTonemapHDRColor"
                    ]


mLossHoleArgument = True
mLossHardArgument = True

dataType = "SS-glossy_shading"

SS_only_ratio = -1

loss_func_name = "Multireso_mLoss"

network_type = "ExtraNet_demodulate_noHistory_SS"