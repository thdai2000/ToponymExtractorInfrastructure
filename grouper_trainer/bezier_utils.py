import bezier # pip install bezier
import numpy as np

import fitCurves

import bezier_curve_fitting

def bezier_from_polyline(sample_pts_x: 'list[float]', sample_pts_y: 'list[float]', max_error = 10) -> 'tuple[list[float], list[float]]':
    # Extract x and y coordinates from sample points
    sample_pts = list(zip(sample_pts_x, sample_pts_y))
    
    # Get control points
    control_points = fitCurves.fitCurve(np.array(sample_pts), max_error)[0]
    
    cpts_x = [pt[0] for pt in control_points]
    cpts_y = [pt[1] for pt in control_points]

    return cpts_x, cpts_y

def bezier_from_polyline_v2(sample_pts_x: 'list[float]', sample_pts_y: 'list[float]') -> 'tuple[list[float], list[float]]':
    control_points = bezier_curve_fitting.get_bezier_parameters(sample_pts_x, sample_pts_y)
    return [pt[0] for pt in control_points], [pt[1] for pt in control_points]

def bezier_to_polyline(control_points_x: 'list[float]', control_points_y: 'list[float]', num_pts: int=8) -> 'tuple[list[float], list[float]]':
    nodes = np.asfortranarray([control_points_x, control_points_y])
    curve = bezier.Curve(nodes, degree=3)
    x, y = curve.evaluate_multi(np.linspace(0, 1, num_pts))
    return list(x), list(y)

def bezier_length(control_points_x: 'list[float]', control_points_y: 'list[float]') -> float:
    nodes = np.asfortranarray([control_points_x, control_points_y])
    curve = bezier.Curve(nodes, degree=3)
    return curve.length

def polyline_length(sample_pts_x: 'list[float]', sample_pts_y: 'list[float]') -> float:
    length = 0
    for i in range(1, len(sample_pts_x)):
        length += np.linalg.norm(np.array([sample_pts_x[i], sample_pts_y[i]]) - np.array([sample_pts_x[i-1], sample_pts_y[i-1]]))
    return length

def _get_bbox_vertices(pts, angle):
    mean = np.float32([pts[:, 0].mean(), pts[:, 1].mean()])
    c, s = np.cos(angle), np.sin(angle)
    R = np.float32([c, -s, s, c]).reshape(2, 2)
    pts = (pts.astype(np.float32) - mean) @ R
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    corners = np.float32([x0, y0, x0, y1, x1, y1, x1, y0])
    corners = corners.reshape(-1, 2) @ R.T + mean
    return corners

from PIL import Image
import cv2

def get_bezier_bbox_params(bezier_pts):
    upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
    lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

    upper_half_x = [p[0] for p in upper_half]
    upper_half_y = [p[1] for p in upper_half]
    lower_half_x = [p[0] for p in lower_half]
    lower_half_y = [p[1] for p in lower_half]

    poly_upper_x, poly_upper_y = bezier_to_polyline(upper_half_x, upper_half_y)
    poly_lower_x, poly_lower_y = bezier_to_polyline(lower_half_x, lower_half_y)

    xs, ys = poly_upper_x + poly_lower_x[::-1], poly_upper_y + poly_lower_y[::-1]

    # Generate non-axis aligned bounding box
    angle = np.arctan2(upper_half_y[0] - upper_half_y[-1], upper_half_x[0] - upper_half_x[-1])

    # Get bbox vertices
    corners_nabb = _get_bbox_vertices(np.array([xs, ys]).T, angle)

    vector_long = corners_nabb[0] - corners_nabb[1]
    vector_short = corners_nabb[1] - corners_nabb[2]

    return corners_nabb, vector_long, vector_short

def get_bezier_bbox(original_image: Image.Image, bezier_pts, scale = 1, reverse_lower = True):
    upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
    lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

    upper_half_x = [p[0] for p in upper_half]
    upper_half_y = [p[1] for p in upper_half]
    lower_half_x = [p[0] for p in lower_half]
    lower_half_y = [p[1] for p in lower_half]

    center = [np.mean(upper_half_x + lower_half_x), np.mean(upper_half_y + lower_half_y)]

    upper_half_x = [scale * (p[0] - center[0]) + center[0] for p in upper_half]
    upper_half_y = [scale * (p[1] - center[1]) + center[1] for p in upper_half]
    lower_half_x = [scale * (p[0] - center[0]) + center[0] for p in lower_half]
    lower_half_y = [scale * (p[1] - center[1]) + center[1] for p in lower_half]

    poly_upper_x, poly_upper_y = bezier_to_polyline(upper_half_x, upper_half_y)
    poly_lower_x, poly_lower_y = bezier_to_polyline(lower_half_x, lower_half_y)

    if reverse_lower:
        xs, ys = poly_upper_x + poly_lower_x[::-1], poly_upper_y + poly_lower_y[::-1]
    else:
        xs, ys = poly_upper_x + poly_lower_x, poly_upper_y + poly_lower_y

    # Generate non-axis aligned bounding box
    udy, udx = upper_half_y[0] - upper_half_y[-1], upper_half_x[0] - upper_half_x[-1]
    ldy, ldx = lower_half_y[0] - lower_half_y[-1], lower_half_x[0] - lower_half_x[-1]
    dy, dx = (udy + ldy) / 2, (udx + ldx) / 2
    angle = np.arctan2(dy, dx)

    return _get_bbox_impl(original_image, xs, ys, angle)

def _get_bbox_impl(original_image: Image.Image, xs, ys, angle):
    '''
    xs, ys: list[float] vertices of convex shape, clockwise order
    '''
    # Plot xs, ys, angle on the image, angle as a line from the center of the shape
    #import matplotlib.pyplot as plt
    #plt.imshow(original_image)
    #plt.plot(xs, ys, 'bo')
    #plt.plot([np.mean(xs)], [np.mean(ys)], 'ro')
    #plt.plot([np.mean(xs), np.mean(xs) + 100 * np.cos(angle)], [np.mean(ys), np.mean(ys) + 100 * np.sin(angle)], 'r')
    #plt.show()

    # Get bbox vertices
    corners_nabb = _get_bbox_vertices(np.array([xs, ys]).T, angle)

    corners_aabb = np.array([corners_nabb.min(axis=0), corners_nabb.max(axis=0)])
    
    center = (corners_aabb[0] + corners_aabb[1]) / 2
    length = np.linalg.norm(corners_aabb[0] - corners_aabb[1])
    corners_aabb[0] = center - length * 1.0 / 2
    corners_aabb[1] = center + length * 1.0 / 2

    # Crop original image
    corners_aabb = corners_aabb.astype(int)
    cropped_image = original_image.crop((corners_aabb[0][0], corners_aabb[0][1], corners_aabb[1][0], corners_aabb[1][1]))

    #plt.imshow(cropped_image)
    #plt.show()

    # Convert nabb to aabb's coordinate system
    corners_nabb -= corners_aabb[0]
    T0 = np.float32([[1, 0, -corners_aabb[0][0]], [0, 1, -corners_aabb[0][1]], [0, 0, 1]])

    # Rotate cropped image and the nabb
    cropped_image = np.array(cropped_image)
    M = cv2.getRotationMatrix2D((cropped_image.shape[1]//2, cropped_image.shape[0]//2), np.degrees(angle) + 180, 1)
    rotated_image = cv2.warpAffine(cropped_image, M, (cropped_image.shape[1], cropped_image.shape[0]))

    corners_nabb = np.hstack([corners_nabb, np.ones((4, 1))]).T

    corners_nabb = M @ corners_nabb

    #plt.imshow(rotated_image)
    #plt.show()

    # Crop the rotated image    
    corners_nabb = corners_nabb.astype(int)

    # Clamp the corners
    corners_nabb[0] = np.clip(corners_nabb[0], 0, rotated_image.shape[1])
    corners_nabb[1] = np.clip(corners_nabb[1], 0, rotated_image.shape[0])

    cropped_rotated_image = rotated_image[corners_nabb[1].min():corners_nabb[1].max(), corners_nabb[0].min():corners_nabb[0].max()]

    # Translation matrix for cropping
    T = np.float32([[1, 0, -corners_nabb[0].min()], [0, 1, -corners_nabb[1].min()], [0, 0, 1]])

    # Combine rotation and translation
    M = T @ np.concatenate((M, np.array([[0, 0, 1]])), axis=0) @ T0

    # Return the image as PIL Image
    return Image.fromarray(cropped_rotated_image), M

    upper_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(0, 8, 2)]
    lower_half = [(bezier_pts[i], bezier_pts[i+1]) for i in range(8, 16, 2)]

    upper_half_x = [p[0] for p in upper_half]
    upper_half_y = [p[1] for p in upper_half]
    lower_half_x = [p[0] for p in lower_half]
    lower_half_y = [p[1] for p in lower_half]

    poly_upper_x, poly_upper_y = bezier_to_polyline(upper_half_x, upper_half_y)
    poly_lower_x, poly_lower_y = bezier_to_polyline(lower_half_x, lower_half_y)

    xs, ys = poly_upper_x + poly_lower_x[::-1], poly_upper_y + poly_lower_y[::-1]

    # Generate non-axis aligned bounding box
    angle = np.arctan2(upper_half_y[0] - upper_half_y[-1], upper_half_x[0] - upper_half_x[-1])

    # Get bbox vertices
    corners_nabb = _get_bbox_vertices(np.array([xs, ys]).T, angle)

    corners_aabb = np.array([corners_nabb.min(axis=0), corners_nabb.max(axis=0)])
    # Expand the bounding box by 10%
    corners_aabb[0] -= 0.1 * (corners_aabb[1] - corners_aabb[0])
    corners_aabb[1] += 0.1 * (corners_aabb[1] - corners_aabb[0])

    # Crop original image
    corners_aabb = corners_aabb.astype(int)
    cropped_image = original_image.crop((corners_aabb[0][0], corners_aabb[0][1], corners_aabb[1][0], corners_aabb[1][1]))

    # Convert nabb to aabb's coordinate system
    corners_nabb -= corners_aabb[0]

    # Rotate cropped image and the nabb
    cropped_image = np.array(cropped_image)
    M = cv2.getRotationMatrix2D((cropped_image.shape[1]//2, cropped_image.shape[0]//2), np.degrees(angle) + 180, 1)
    rotated_image = cv2.warpAffine(cropped_image, M, (cropped_image.shape[1], cropped_image.shape[0]))

    corners_nabb = np.hstack([corners_nabb, np.ones((4, 1))]).T

    corners_nabb = M @ corners_nabb

    # Crop the rotated image    
    corners_nabb = corners_nabb.astype(int)

    # Clamp the corners
    corners_nabb[0] = np.clip(corners_nabb[0], 0, rotated_image.shape[1])
    corners_nabb[1] = np.clip(corners_nabb[1], 0, rotated_image.shape[0])

    cropped_rotated_image = rotated_image[corners_nabb[1].min():corners_nabb[1].max(), corners_nabb[0].min():corners_nabb[0].max()]

    # Return the image as PIL Image
    return Image.fromarray(cropped_rotated_image)