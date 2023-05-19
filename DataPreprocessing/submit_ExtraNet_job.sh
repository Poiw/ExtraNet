#!/bin/sh
#SBATCH -c 32

# python preprocess.py /export/work/songyin/Infiltrator0430/Seq6
python compressData.py

# python validate_extraSS.py --extraSS_model_path /export/work/songyin/Log/Bunker/2023-04-16-10-23-41__Bunker-TrainAll-Length16_Joint-SS-ExtraSS-OF-DirectNet_maskedUpscale2_ex2/models/checkpoint.pth --output_dir /export/work/songyin/Test/Bunker_Train1_Direct_0417_Length16_noShadowSeparation_OFNet-maskUpscale2_seq1-250-370
# python validate_extraSS_single.py --model_path /export/work/songyin/Log/RedwoodForest/2023-05-16-22-16-58__RedwoodForest-TrainAll-Length16_Single-SS-ExtraSS-OF-DirectNet-v2-noUpscale_BZ12_ex4_newbilinear/models/checkpoint.pth --output_dir /export/work/songyin/Test/RedwoodForest_Seq5-1240-1480_0517_newWarp
# python makeData_new.py
# python cropData.py --output_dir /export/work/songyin/Sequencer/CroppedData/Seq7
# python his_train.py --bz 8 --info Bunker_his_OFNetSimple_RawReso
# python joint_recurrent_train_singleModel.py --bz 12 --lr 0.0004 --num_works 14 --info Test