import numpy as np
import torch
import config

normal_hole_threshold = 0.98
worldPosition_hole_threshold_params = (7.5, 45, 50)


def lerp(a, b, alpha):
    return a * alpha + b * (1 - alpha)

def merge_by_max(matrix0, matrix1):
    res = np.zeros([matrix0.shape[0], matrix0.shape[1]]).astype(np.float32)
    res[matrix0 > matrix1] = matrix0[matrix0 > matrix1]
    res[matrix1 >= matrix0] = matrix1[matrix1 >= matrix0]
    return res

def min_world_pos_distance(worldPos_previous, worldPos_now, height, width, warp_list):

    warp2_i0, warp2_i1, warp2_j0, warp2_j1 = warp_list

    warp_i0j0 = worldPos_previous[warp2_i0, warp2_j0]
    warp_i0j1 = worldPos_previous[warp2_i0, warp2_j1]
    warp_i1j0 = worldPos_previous[warp2_i1, warp2_j0]
    warp_i1j1 = worldPos_previous[warp2_i1, warp2_j1]

    warp_i0j0 = warp_i0j0.reshape((height, width, 3)).astype(np.float32)
    warp_i0j1 = warp_i0j1.reshape((height, width, 3)).astype(np.float32)
    warp_i1j0 = warp_i1j0.reshape((height, width, 3)).astype(np.float32)
    warp_i1j1 = warp_i1j1.reshape((height, width, 3)).astype(np.float32)

    distance00 = np.sqrt(np.square(warp_i0j0 - worldPos_now).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(np.float32)
    distance01 = np.sqrt(np.square(warp_i0j1 - worldPos_now).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(np.float32)
    distance10 = np.sqrt(np.square(warp_i1j0 - worldPos_now).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(np.float32)
    distance11 = np.sqrt(np.square(warp_i1j1 - worldPos_now).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(np.float32)

    res = merge_by_max(distance00, distance01)
    res = merge_by_max(res, distance10)
    res = merge_by_max(res, distance11)
    return res


def Linear_Warp(img, height, width, warp_list, weight_list):

    warp2_i0, warp2_i1, warp2_j0, warp2_j1 = warp_list
    weight2_i0, weight2_i1, weight2_j0, weight2_j1 = weight_list

    res2_i0j0 = img[warp2_i0, warp2_j0]
    res2_i1j0 = img[warp2_i1, warp2_j0]
    res2_i0j1 = img[warp2_i0, warp2_j1]
    res2_i1j1 = img[warp2_i1, warp2_j1]

    res2_j0 = res2_i0j0 * weight2_i0 + res2_i1j0 * weight2_i1
    res2_j1 = res2_i0j1 * weight2_i0 + res2_i1j1 * weight2_i1
    res2 = res2_j0 * weight2_j0 + res2_j1 * weight2_j1
    res2 = res2.reshape((height, width, img.shape[2])).astype(img.dtype)

    return res2


def cal_warp_index_weight(warp_index, max_height, max_width):
    warp_i, warp_j = warp_index

    warp_i0 = np.floor(warp_i)
    warp_i1 = np.floor(warp_i) + 1
    warp_j0 = np.floor(warp_j)
    warp_j1 = np.floor(warp_j) + 1

    weight_i0 = (1 - np.abs(warp_i0 - warp_i))[:, np.newaxis]
    weight_i1 = (1 - np.abs(warp_i1 - warp_i))[:, np.newaxis]
    weight_j0 = (1 - np.abs(warp_j0 - warp_j))[:, np.newaxis]
    weight_j1 = (1 - np.abs(warp_j1 - warp_j))[:, np.newaxis]

    # clamp out of screen indexes
    warp_i0 = np.clip(warp_i0, 0, max_height + 1).astype(np.long)
    warp_i1 = np.clip(warp_i1, 0, max_height + 1).astype(np.long)
    warp_j0 = np.clip(warp_j0, 0, max_width  + 1).astype(np.long)
    warp_j1 = np.clip(warp_j1, 0, max_width  + 1).astype(np.long)

    return [weight_i0, weight_i1, weight_j0, weight_j1], [warp_i0, warp_i1, warp_j0, warp_j1]


def GetOCCMV(motion2, depthCurr):
    # warp the motion vector back so that we can use occ motion vector
    height, width, _ = motion2.shape

    occmv = np.zeros((height, width, _))

    cullingBuffer = np.zeros((height, width))
    cullingBuffer[cullingBuffer == 0.] = 10.
    # back warp scattering
    for i in range(height):
        for j in range(width):
            samplePos = np.array([i + motion2[i, j, 1], j - motion2[i, j, 0]])
            # check four neighbour, overwrite them if the len of this mv is bigger than the other
            Pos00 = np.floor(samplePos).astype(np.int)
            Pos10 = np.array([Pos00[0] + 1, Pos00[1]])
            Pos01 = np.array([Pos00[0], Pos00[1] + 1])
            Pos11 = np.array([Pos00[0] + 1, Pos00[1] + 1])
            if Pos00[0] < 0 or Pos00[0] >= height or Pos00[1] < 0 or Pos00[1] >= width:
                pass
            else:
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos00[0], Pos00[1]]) :
                if cullingBuffer[Pos00[0], Pos00[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos00[0], Pos00[1]] = depthCurr[i, j][0]
                    occmv[Pos00[0], Pos00[1]] = motion2[i, j]
            if Pos01[0] < 0 or Pos01[0] >= height or Pos01[1] < 0 or Pos01[1] >= width:
                pass
            else:
                # if np.square(motion2[i, j]).sum() > np.square(occmv[Pos01[0], Pos01[1]]).sum():
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos01[0], Pos01[1]]) :
                if cullingBuffer[Pos01[0], Pos01[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos01[0], Pos01[1]] = depthCurr[i, j][0]
                    occmv[Pos01[0], Pos01[1]] = motion2[i, j]
            if Pos10[0] < 0 or Pos10[0] >= height or Pos10[1] < 0 or Pos10[1] >= width:
                pass
            else:
                # if np.square(motion2[i, j]).sum() > np.square(occmv[Pos10[0], Pos10[1]]).sum():
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos10[0], Pos10[1]]) :
                if cullingBuffer[Pos10[0], Pos10[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos10[0], Pos10[1]] = depthCurr[i, j][0]
                    occmv[Pos10[0], Pos10[1]] = motion2[i, j]
            if Pos11[0] < 0 or Pos11[0] >= height or Pos11[1] < 0 or Pos11[1] >= width:
                pass
            else:
                # if np.square(motion2[i, j]).sum() > np.square(occmv[Pos11[0], Pos11[1]]).sum():
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos11[0], Pos11[1]]) :
                if cullingBuffer[Pos11[0], Pos11[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos11[0], Pos11[1]] = depthCurr[i, j][0]
                    occmv[Pos11[0], Pos11[1]] = motion2[i, j]
    # final mv generate
    finalmv = np.zeros((height, width, _))

    flat_index = np.arange(height * width)
    i = flat_index // width
    j = flat_index - i * width

    warp2_i = np.clip(np.floor(i + motion2[i, j, 1]), 0, height - 1).astype(np.long)
    warp2_j = np.clip(np.floor(j - motion2[i, j, 0]), 0, width - 1).astype(np.long)

    finalmv = occmv[warp2_i, warp2_j]
    finalmv = finalmv.reshape((height, width, 3)).astype(np.float32)
    
    
    return finalmv


def warp_img(img, motion_vector):

    height, width = img.shape[0], img.shape[1]

    img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), constant_values=-10.0)
    mv_padded = np.pad(motion_vector, ((1, 1), (1, 1), (0, 0)), constant_values=0)


    flat_index = np.arange(height*width)
    i = flat_index // width
    j = flat_index - i * width

    # Because we will pad original image in warp
    i += 1
    j += 1

    # motion vector record the change in SCREEN SPACE! and indexed by XY!
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑ → → → → → → → →
    # when we index the array by i, j the space is like that
    # ↓ → → → → → → → →
    # ↓
    # ↓
    # ↓
    # ↓
    # ↓
    # ↓
    # keep in mind with that

    warp_i = i + mv_padded[i, j, 1]
    warp_j = j - mv_padded[i, j, 0]

    weight_list, warp_list = cal_warp_index_weight([warp_i, warp_j], height, width)

    warpped_img = Linear_Warp(img_padded, height, width, warp_list, weight_list)

    warpped_img[warpped_img < 0] = -1

    return warpped_img


def warp_img_with_hole(img, motion_vector, normal_pair, stencil_pair, worldPosition_pair, depth, NoV):

    height, width = img.shape[0], img.shape[1]

    normal_previous, normal_now = normal_pair
    stencil_previous, stencil_now = stencil_pair
    worldPosition_previous, worldPosition_now = worldPosition_pair

    normal_previous = np.pad(normal_previous, ((1, 1), (1, 1), (0, 0)), constant_values=0)
    stencil_previous = np.pad(stencil_previous, ((1, 1), (1, 1), (0, 0)), constant_values=0)
    worldPosition_previous = np.pad(worldPosition_previous, ((1, 1), (1, 1), (0, 0)), constant_values=1e5)

    img_padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), constant_values=-10.0)
    mv_padded = np.pad(motion_vector, ((1, 1), (1, 1), (0, 0)), constant_values=0)


    flat_index = np.arange(height*width)
    i = flat_index // width
    j = flat_index - i * width

    # Because we will pad original image in warp
    i += 1
    j += 1

    # motion vector record the change in SCREEN SPACE! and indexed by XY!
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑
    # ↑ → → → → → → → →
    # when we index the array by i, j the space is like that
    # ↓ → → → → → → → →
    # ↓
    # ↓
    # ↓
    # ↓
    # ↓
    # ↓
    # keep in mind with that

    warp_i = i + mv_padded[i, j, 1]
    warp_j = j - mv_padded[i, j, 0]

    weight_list, warp_list = cal_warp_index_weight([warp_i, warp_j], height, width)

    warpped_img = Linear_Warp(img_padded, height, width, warp_list, weight_list)
    warpped_img[warpped_img < 0] = -1

    normal_warp = Linear_Warp(normal_previous, height, width, warp_list, weight_list)
    stencil_warp = Linear_Warp(stencil_previous, height, width, warp_list, weight_list)


     # moving actor
    normal_warp_length = np.sqrt(np.square(normal_warp).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)
    normal_now_length = np.sqrt(np.square(normal_now).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)


    normal_warp /= (normal_warp_length + 1e-5)
    cos_normal = (normal_warp * normal_now).sum(axis=-1, keepdims=True).repeat(3, axis=-1)
    # normal difference
    normal_difference = cos_normal < normal_hole_threshold
    # false only when both are zero
    empty = np.logical_or(normal_warp_length != 0, normal_now_length != 0)
    normal_difference = np.logical_and(normal_difference, empty)
    b_moving_actor = stencil_warp != 0
    moving_actor_hole = np.logical_and(normal_difference, b_moving_actor)

    # static actor
    bias = lerp(worldPosition_hole_threshold_params[0], worldPosition_hole_threshold_params[1], np.abs(NoV[:, :, 0])) + depth[:, :, 0] * worldPosition_hole_threshold_params[2]
    world_position_distance = min_world_pos_distance(worldPosition_previous, worldPosition_now, height, width, warp_list)
    world_position_diff = (world_position_distance > bias).repeat(3, axis=-1).reshape(height, width, 3)
    b_static_actor = stencil_warp == 0
    static_disocc = np.logical_and(world_position_diff, b_static_actor)

    # hole for moving actor
    mesh_minus = np.abs(stencil_warp - stencil_now).sum(axis=-1, keepdims=True).repeat(3, axis=-1)
    mesh_hole = mesh_minus > 1

    # hole = np.logical_or(normal_difference, disocc, mesh_diff)
    hole = np.logical_or(static_disocc, mesh_hole)
    hole = np.logical_or(hole, moving_actor_hole)

    return warpped_img, hole



Upsampler2_Nearest = torch.nn.Upsample(scale_factor=2)

def Process_Input(data, type="color", ss_only_flag=False):

    extras = {}

    # if type == "color":

    #     occWarp_img = data["occ-warp_PreTonemapHDRColor"]
    #     depth = data["SceneDepth"]
    #     metallic = data["Metallic"]
    #     roughness = data["Roughness"]
    #     basecolor = data["BaseColor"]
    #     normal = data["WorldNormal"]
    #     specular = data["Specular"]

    #     input = torch.cat([occWarp_img, depth, metallic, roughness, basecolor, normal, specular], dim=1)

    #     mask = torch.zeros_like(occWarp_img)
    #     mask[occWarp_img < 0] = 0
    #     mask[occWarp_img >= 0] = 1

    #     extras["mask"] = mask

    #     gt = data["PreTonemapHDRColor"]

    #     input = input.cuda()
    #     gt = gt.cuda()

    # elif type == "glossy_shading":
    #     occWarp_demodulate_img = data["occ-warp_demodulatePreTonemapHDRColor"]
    #     depth = data["SceneDepth"]
    #     metallic = data["Metallic"]
    #     roughness = data["Roughness"]
    #     basecolor = data["BaseColor"]
    #     normal = data["WorldNormal"]
    #     specular = data["Specular"]

    #     input = torch.cat([occWarp_demodulate_img, depth, metallic, roughness, basecolor, normal, specular], dim=1)

    #     mask = torch.zeros_like(occWarp_demodulate_img)
    #     mask[occWarp_demodulate_img < 0] = 0
    #     mask[occWarp_demodulate_img >= 0] = 1

    #     extras["mask"] = mask

    #     gt = data["demodulatePreTonemapHDRColor"]
        
    #     input = input.cuda()
    #     gt = gt.cuda()

    # elif type == "SS-glossy_shading":

    #     # mixed training ss only and extra+ss
    #     if config.SS_only_ratio > 0:
    #         if np.random.rand() < config.SS_only_ratio:
    #             data["occ-warp_demodulatePreTonemapHDRColor"] = data["demodulatePreTonemapHDRColor"]
    #     if ss_only_flag:
    #             data["occ-warp_demodulatePreTonemapHDRColor"] = data["demodulatePreTonemapHDRColor"]

    #     occWarp_demodulate_img = data["occ-warp_demodulatePreTonemapHDRColor"]

    #     depth = data["SceneDepth"]
    #     metallic = data["Metallic"]
    #     roughness = data["Roughness"]
    #     basecolor = data["BaseColor"]
    #     normal = data["WorldNormal"]
    #     specular = data["Specular"]

    #     low_input = torch.cat([occWarp_demodulate_img, basecolor, metallic, specular, depth, roughness, normal], dim=1).cuda()
    #     high_input = data["occ-warp_HighResoTAAPreTonemapHDRColor"].cuda()

    #     mask = torch.zeros_like(occWarp_demodulate_img)
    #     mask[occWarp_demodulate_img < 0] = 0
    #     mask[occWarp_demodulate_img >= 0] = 1

    #     extras["mask"] = mask
        
    #     high_mask = torch.zeros_like(high_input)
    #     high_mask[high_input < 0] = 0
    #     high_mask[high_input >= 0] = 1

    #     extras["high_mask"] = high_mask

    #     gt = {"low" : data["PreTonemapHDRColor"].cuda(), "high" : data["HighResoTAAPreTonemapHDRColor"].cuda()}
    #     input = {"low" : low_input, "high" : high_input}

    if type == "SS-glossy_shading":

        # mixed training ss only and extra+ss
        if config.SS_only_ratio > 0:
            if np.random.rand() < config.SS_only_ratio:
                data["occ-warp_demodulatePreTonemapHDRColor"] = data["demodulatePreTonemapHDRColor"]
        if ss_only_flag:
                data["occ-warp_demodulatePreTonemapHDRColor"] = data["demodulatePreTonemapHDRColor"]

        occWarp_demodulate_img = data["occ-warp_demodulatePreTonemapHDRColor"]
        warp_demodulate_img = data["warp_demodulatePreTonemapHDRColor"]

        depth = data["SceneDepth"]
        metallic = data["Metallic"]
        roughness = data["Roughness"]
        basecolor = data["BaseColor"]
        normal = data["WorldNormal"]
        specular = data["Specular"]

        low_input = torch.cat([occWarp_demodulate_img, basecolor, metallic, specular, depth, roughness, normal], dim=1).cuda()
        high_input = data["occ-warp_HighResoTAAPreTonemapHDRColor"].cuda()

        mask = torch.zeros_like(warp_demodulate_img)
        mask[warp_demodulate_img < 0] = 0
        mask[warp_demodulate_img >= 0] = 1

        extras["mask"] = mask
        
        high_mask = Upsampler2_Nearest(mask)

        extras["high_mask"] = high_mask
        
        data["mask"] = mask
        data["high_mask"] = high_mask

        gt = {"low" : data["PreTonemapHDRColor"].cuda(), "high" : data["HighResoTAAPreTonemapHDRColor"].cuda()}
        
        input = {"low" : low_input, "high" : high_input}

    else:
        raise NotImplementedError

    return input, extras, gt

def Postprocess(data, pred, type="color"):

    with torch.no_grad():


        if type == "color":
            pred = pred.detach().cpu()
            data["pred"] = pred
            pass

        elif type == "glossy_shading":
            
            pred = pred.detach().cpu()
            data["pred"] = pred

            img_albedo = data["BaseColor"] + data["Specular"] * 0.08 * ( 1-data["Metallic"] )
            img_albedo = img_albedo.detach().cpu()
            pred_color = img_albedo * pred

            data["pred_color"] = pred_color

        elif type == "SS-glossy_shading":

            data["low_pred"] = pred["low"].detach().cpu()
            data["high_pred"] = pred["high"].detach().cpu()

        else:
            raise NotImplementedError

    return data

def getL1(pred, data, type="color"):

    if type == "SS-glossy_shading":
        low_pred = pred["low"].detach().cpu()
        high_pred = pred["high"].detach().cpu()

        low_gt = data["PreTonemapHDRColor"].detach().cpu()
        high_gt = data["HighResoTAAPreTonemapHDRColor"].detach().cpu()

        return torch.abs(low_pred-low_gt).mean(), torch.abs(high_pred-high_gt).mean()

    else:
        raise NotImplementedError

