import gc
import math
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path

import cv2
import kornia as K
import numpy as np

import torch
import torch.nn as nn
from PIL import Image
from kornia.feature import LoFTR
from pyproj import Transformer
from rasterio.transform import rowcol, xy
from transformers import AutoImageProcessor, AutoModel

from utils.parser import parse_txt_gbk
from utils.geo_processing import wgs842ecef, calculate_ground_coordinates, query_DEM, crop_basemap_from_pyramid

from utils.core_utils import timer, rotate_points, rotate_by_90_multiple, get_affine_transform, save_images
from utils.viz import draw_matches_fixed_width_centered, draw_loftr_matches


@timer
def coarse_alignment(image1, image2, angles):
    print(f'{sys._getframe().f_code.co_name} processing...')

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    ckpt_path = (Path(__file__).parent / 'ckpt/dinov3-vitl16-pretrain-sat493m').as_posix()

    processor = AutoImageProcessor.from_pretrained(ckpt_path)
    model = AutoModel.from_pretrained(ckpt_path).to(device)

    with torch.no_grad():
        inputs1 = processor(images=image1, return_tensors="pt").to(device)
        outputs1 = model(**inputs1)
        pooled_output1 = outputs1.pooler_output

    results = []
    for ang in angles:
        rot = rotate_by_90_multiple(image2, ang)
        pil_img_rot = Image.fromarray(cv2.cvtColor(rot, cv2.COLOR_BGR2RGB))

        with torch.no_grad():
            inputs2 = processor(images=pil_img_rot, return_tensors="pt").to(device)
            outputs2 = model(**inputs2)
            pooled_output2 = outputs2.pooler_output

        cos = nn.CosineSimilarity(dim=0)
        sim2 = cos(pooled_output1[0], pooled_output2[0]).item()
        sim = (sim2 + 1) / 2
        results.append((ang, sim))
        print(f'Angle={ang:.1f}°, Cosine Similarity={sim:.3f}')

    arr = np.array(results)
    theta = angles[np.argmax(arr[:, 1])]

    return theta


def get_rotation_angle(path, record, pts3d_geo, x_res, y_res, pyramid_path, top_layer, best_level, best_res, tile_size,
                       **kwargs):
    img = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]  # 3794, 4744
    dw, dh = x_res * w, y_res * h
    center_lon, center_lat = calculate_ground_coordinates(*pts3d_geo, record['机下点海拔高度(米)'],
                                                          record['视场中心俯仰角(度)'], record['视场中心横滚角(度)'])

    cropped, _ = crop_match_basemap(center_lon, center_lat, dw, dh, pyramid_path, top_layer, best_level,
                                    best_res, tile_size, keep_square=True, crop_scale=kwargs['crop_scale'])
    resized = cv2.resize(img, None, fx=x_res / best_res, fy=y_res / best_res)

    candidate_angles = list(range(0, 360, 90))
    theta = coarse_alignment(cropped, resized, candidate_angles)  # 也可以直接送入原图，但resized更快
    rotated = rotate_by_90_multiple(resized, theta)
    canvas = draw_matches_fixed_width_centered(cropped, rotated)
    save_path = f"results/rot_{theta}_{path.stem}.jpg"
    Path(save_path).resolve().parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, canvas)
    print(f'rotation angle: {theta}\n')

    return theta


def validate_quad_geometry(n_inlier, inlier_ratio, prj_error, corners, max_area=100, min_angle=30, max_angle=150,
                           max_aspect_ratio=4.0, min_inliers=20, min_ratio=0.05):
    failed = True if n_inlier < min_inliers or inlier_ratio < min_ratio else False
    if failed:
        return False, f"Not enough inliers: {n_inlier} < 20 or inlier ratio: {inlier_ratio:.2f} < 0.05"

    # 1. 格式标准化
    pts = np.array(corners, dtype=np.float32).reshape(4, 2)

    # 2. 凸性检查 (Convexity Check)
    # cv2.isContourConvex 需要 int32 类型
    if not cv2.isContourConvex(pts.astype(np.int32)):
        return False, "Shape is not convex (Self-intersecting)"

    # 3. 面积检查 (Area Check)
    area = cv2.contourArea(pts)
    if area > max_area:
        return False, f"Area too large: {area:.1f} < {max_area}"

    # 4. 边长计算
    # 计算四条边的向量和长度
    # v0: P0->P1, v1: P1->P2, v2: P2->P3, v3: P3->P0
    vectors = []
    lengths = []
    for i in range(4):
        p_curr = pts[i]
        p_next = pts[(i + 1) % 4]
        vec = p_next - p_curr
        length = np.linalg.norm(vec)

        vectors.append(vec)
        lengths.append(length)

    # 5. 边长比例检查 (Aspect Ratio / Distortion)
    # 如果最长边是最短边的 10 倍，说明配准严重拉伸
    min_len = min(lengths)
    max_len = max(lengths)

    if min_len == 0:
        return False, "Edge length is zero (Degenerate points)"

    ratio = max_len / min_len
    if ratio > max_aspect_ratio:
        return False, f"Aspect ratio too high: {ratio:.1f} (Extreme stretching)"

    # 6. 内角检查 (Angle Check)
    # 正常视角下，矩形投影后的内角应该接近 90 度
    angles = []
    for i in range(4):
        # 获取相邻的两个向量
        # 当前角点是 i，进入向量是 v_{i-1}，离开向量是 v_{i}
        # 为了计算内角，我们需要两个从点 i 发出的向量
        # 向量 A: P_i -> P_{i+1} (即 vectors[i])
        # 向量 B: P_i -> P_{i-1} (即 -vectors[i-1])

        v_a = vectors[i]
        v_b = -vectors[(i - 1) % 4]

        # 计算夹角公式: cos(theta) = (a . b) / (|a|*|b|)
        len_a = lengths[i]
        len_b = lengths[(i - 1) % 4]

        if len_a * len_b == 0: continue  # 应该已经被 min_len 拦截了

        cos_theta = np.dot(v_a, v_b) / (len_a * len_b)
        # 截断以防浮点误差超出 [-1, 1]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)

        angle_deg = math.degrees(math.acos(cos_theta))
        angles.append(angle_deg)

        if angle_deg < min_angle:
            return False, f"Corner {i} too sharp: {angle_deg:.1f}°"
        if angle_deg > max_angle:
            return False, f"Corner {i} too flat: {angle_deg:.1f}°"

    # 通过所有检查
    return True, "Valid"


def loftr_match(src_img, dst_img, filename='', **kwargs):
    loftr = LoFTR('outdoor')
    matcher = loftr.eval().cuda() if torch.cuda.is_available() else loftr.eval()

    # 读取两张图 → gray tensor [0,1] 1×1×H×W, float32
    def load_torch(img):
        if len(img.shape) == 3:  # 彩色图
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = K.image_to_tensor(img, keepdim=False).float() / 255.

        return img  # 1×1×H×W

    src_t = load_torch(src_img)
    dst_t = load_torch(dst_img)

    with torch.no_grad():
        input_dict = {'image0': src_t.cuda(), 'image1': dst_t.cuda()}
        match_res = matcher(input_dict)

    mkpts1 = match_res['keypoints0'].cpu().numpy()
    mkpts2 = match_res['keypoints1'].cpu().numpy()
    conf = match_res['confidence'].cpu().numpy()
    n_pts = len(mkpts1)
    if n_pts < 4:
        return None

    reproj_thresh = kwargs.get('reproj_thresh', 5.0)  # >7时很容易误匹配
    homo, mask = cv2.findHomography(mkpts1, mkpts2, cv2.RANSAC, ransacReprojThreshold=reproj_thresh)
    mask = mask.ravel().astype(bool)
    pts1_in = mkpts1[mask]
    pts2_in = mkpts2[mask]
    mconf = conf[mask]
    n_inlier = pts1_in.shape[0]
    inlier_ratio = n_inlier / n_pts
    print(f'LoFTR 原始: {n_pts} 对 → RANSAC 内点: {mask.sum()} 对  ratio: {inlier_ratio:.2f}')

    pts1 = pts1_in.reshape(-1, 1, 2)
    pts2 = pts2_in.reshape(-1, 1, 2)
    errors = np.linalg.norm(pts2 - cv2.perspectiveTransform(pts1, homo), axis=2)
    median_error = np.median(errors)
    stats = n_inlier, inlier_ratio, median_error

    h, w = src_img.shape[:2]
    corners_src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32).reshape(-1, 1, 2)
    corners_dst = cv2.perspectiveTransform(corners_src, homo)
    base_area = dst_img.shape[0] * dst_img.shape[1]
    valid, message = validate_quad_geometry(
        n_inlier, inlier_ratio, median_error,
        corners_dst,
        max_area=base_area,  # 最小面积，根据底图分辨率调整
        min_angle=45,  # 允许最小 45 度 (稍微严格一点)
        max_angle=135,  # 允许最大 135 度
        max_aspect_ratio=3.0,  # 长宽比不超过 3 倍
        min_inliers=kwargs['min_inliers'],
        min_ratio=kwargs['min_ratio']
    )

    text = f"Matches: {n_inlier}\nInlier ratio: {inlier_ratio:.3f}\nreproj err: {median_error:.2f}"
    vis = draw_loftr_matches(src_img, dst_img, homo, text, pts1_in, pts2_in, mconf, filename, valid)

    gc.collect()
    torch.cuda.empty_cache()

    if not valid:
        return None

    return pts1_in, pts2_in, homo, stats, vis


def crop_match_basemap(center_lon, center_lat, dw, dh, pyramid_path, top_layer, tar_layer, tar_resolution,
                       tar_tile_size, crop_scale=1.5, keep_square=True):
    dw, dh = dw * crop_scale, dh * crop_scale
    if keep_square:  # 光轴倾斜参数在南北方向不稳定，推荐裁剪时保持正方形，留一定的缓冲余量
        dw = dh = (dw + dh) / 2

    cropped, transform = crop_basemap_from_pyramid(pyramid_path, center_lon, center_lat, dw, dh, top_layer, tar_layer,
                                                   tar_resolution, tar_tile_size)
    return cropped, transform


@timer
def match_one_image(rotated, cropped, transform, dem_file, filename, **kwargs):
    results = loftr_match(rotated, cropped, filename, **kwargs)
    if results is None:
        return None

    pts0, pts1, homography, stats, vis = results
    row, col = pts1[:, 1], pts1[:, 0]
    xs, ys = xy(transform, row, col)  # 像素转投影坐标

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    pts3d_geo = transformer.transform(xs, ys)
    pts3d_geo = np.array(pts3d_geo).T
    alts = query_DEM(dem_file, pts3d_geo)

    pts3d_ecef = wgs842ecef(np.column_stack([pts3d_geo, alts]))

    return pts0, pts3d_ecef, pts1, homography, stats, vis


def generate_matched_points(img, record, pts3d_geo, x_res, y_res, theta,
                            pyramid_path, top_layer, best_level, best_res, tile_size,
                            dem_file, purename, **kwargs):
    fx, fy = x_res / best_res, y_res / best_res
    resized = cv2.resize(img, None, fx=fx, fy=fy)  # 缩放到底图分辨率
    rotated = rotate_by_90_multiple(resized, theta)

    h, w = img.shape[:2]  # 3794, 4744
    affine, size_t = get_affine_transform(w, h, fx, fy, - theta)

    dw, dh = x_res * w, y_res * h  # 需要裁剪的地面范围（米）
    if theta == 90 or theta == 270:
        dh, dw = dw, dh

    center_lon, center_lat = calculate_ground_coordinates(*pts3d_geo, record['机下点海拔高度(米)'],
                                                          record['视场中心俯仰角(度)'], record['视场中心横滚角(度)'])

    cropped, transform = crop_match_basemap(center_lon, center_lat, dw, dh, pyramid_path, top_layer,
                                            best_level, best_res, tile_size, crop_scale=kwargs['crop_scale'])

    results = match_one_image(rotated, cropped, transform, dem_file, purename, **kwargs)
    if results is None:
        return None

    pts_2d, pts_3d_ecef, pts2d_sat, homo_rot2sat, stats, vis = results
    pts2d_rotated = rotate_points(pts_2d, 360 - theta, *rotated.shape)
    xs, ys = xy(transform, pts2d_sat[:, 1], pts2d_sat[:, 0])  # 像素转投影坐标
    pts2d_src = pts2d_rotated / (fx, fy)

    home_src2rot = np.vstack([affine, [0, 0, 1]])
    homo_src2sat = homo_rot2sat @ home_src2rot

    return pts2d_src, pts_3d_ecef, pts2d_sat, homo_src2sat, xs, ys, rotated, stats, vis
