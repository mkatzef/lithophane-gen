import argparse
import numpy as np
from stl import mesh
import imageio
from PIL import Image

arr = np.array


def get_tr_cones(origin, step, up, diag, right):
    """
    Returns a set of 4 triangles ; 2 making a square base and 2 giving the top (texture) layer.
    """
    ox, oy, oz = origin
    triplets = [
        [
            [ox+step, oy+step, diag],
            [ox, oy+step, up],
            [ox, oy, oz],
        ],
        [
            [ox, oy, oz],
            [ox+step, oy, right],
            [ox+step, oy+step, diag],
        ],
        [
            [ox, oy, 0],
            [ox, oy+step, 0],
            [ox+step, oy+step, 0]
        ],
        [
            [ox+step, oy+step, 0],
            [ox+step, oy, 0],
            [ox, oy, 0],
        ],
    ]

    return triplets


def get_tr_cones_circ(base_here, base_up, base_diag, base_right, val_here, val_up, val_diag, val_right):
    """
    Returns 2 base triangles and 2 texture triangles as one cell of a circular stl
    """
    # Close to zero, treat as empty
    if (abs(arr([val_here-base_here, val_up-base_up, val_diag-base_diag, val_right-base_right])) < 1e-3).all():
        return []
    return [
        [val_here, val_up, val_diag],
        [val_diag, val_right, val_here],
        [base_here, base_right, base_diag],
        [base_diag, base_up, base_here]
    ]


def to_gray(rgb):
    """
    Color to grayscale
    """
    return np.dot(rgb[... , :3] , [0.299 , 0.587, 0.114])


def img_to_planar_mesh(depth_pixels, width_mm):
    """
    2D lithograph gen
    """
    # add border of zeros
    n_rows = len(depth_pixels) + 2
    n_cols = len(depth_pixels[0]) + 2
    tmp = np.zeros((n_rows, n_cols))
    tmp[1:-1, 1:-1] = depth_pixels
    depth_pixels = tmp

    mm_per_cell = width_mm / (n_cols - 1)
    height_mm = mm_per_cell * n_rows

    trips_list = []
    next_row = None
    for row in range(n_rows-1):
        if next_row is None:
            current_row = depth_pixels[row,:]
        else:
            current_row = next_row
        next_row = depth_pixels[row+1,:]

        for col, val in enumerate(current_row[:-1]):
            start = ((col+0.5) * mm_per_cell, (row+0.5) * mm_per_cell, val)
            up = next_row[col]
            diag = next_row[col+1]
            right = current_row[col+1]
            trips_list += get_tr_cones(start, mm_per_cell, up, diag, right)

    return trips_list


def img_to_circular_mesh(depth_pixels, radius_mm, max_depth_mm, base_height_mm=16):
    """
    Circular lithograph gen
    """
    depth_pixels = np.fliplr(depth_pixels)

    # add border of zeros
    n_cols = len(depth_pixels[0]) + 1
    n_depth_rows = len(depth_pixels)
    rads_per_cell = 2*np.pi / (n_cols-1)
    cell_height_mm = 2*np.pi*radius_mm / n_cols
    n_base_rows = int(base_height_mm / cell_height_mm)

    n_rows = n_depth_rows + n_base_rows + 2
    height_mm = n_rows * cell_height_mm

    tmp = np.zeros((n_rows, n_cols))
    tmp[n_base_rows+1:-1, :-1] = depth_pixels
    tmp[n_base_rows+1:-1, -1] = depth_pixels[:,0]
    depth_pixels = tmp

    if base_height_mm > 0:
        # add base border
        n_ridge_changes = 7  # keep odd
        n_divot_holes = 6
        divot_hole_width_rads = np.radians(15)
        pixels_per_hole = int(divot_hole_width_rads / (2*np.pi) * n_cols)
        pixels_between_hole_starts = int(n_cols / n_divot_holes)
        cells_per_ridge = n_base_rows // n_ridge_changes
        base_offset = n_base_rows - n_ridge_changes * cells_per_ridge + 1
        depth_pixels[1:base_offset, :] = max_depth_mm
        shifted = False
        had_left_divot = False
        for i in range(base_offset, n_base_rows):
            if ((i-base_offset)//cells_per_ridge) % 2:
                # divot
                if had_left_divot:
                    shifted = not shifted
                    had_left_divot = False
                depth_pixels[i, :] = max_depth_mm - 1
                for j in range(n_divot_holes):
                    hole_start = int((j+shifted/2) * pixels_between_hole_starts)
                    depth_pixels[i, max(1, hole_start) : min(n_cols-1, hole_start+pixels_per_hole)] = 0
            else:
                # ridge
                depth_pixels[i, :] = max_depth_mm
                had_left_divot = True

    unit_vecs = [arr([np.sin(i*rads_per_cell), np.cos(i*rads_per_cell)]) for i in range(n_cols)]
    vecs_img = np.repeat(arr(unit_vecs).reshape((1,-1,2)), n_rows, axis=0)
    base_vecs_img = radius_mm * vecs_img
    tex_vecs_img = (radius_mm + depth_pixels)[:, :, np.newaxis] * vecs_img

    z_img = arr([[r/n_rows * height_mm] * n_cols for r in range(n_rows)])
    z_img = np.expand_dims(z_img, axis=2)
    base_vecs_img = np.concatenate((base_vecs_img, z_img), axis=2)
    tex_vecs_img = np.concatenate((tex_vecs_img, z_img), axis=2)

    trips_list = []
    for row in range(n_rows-1):
        for col in range(n_cols-1):
            base_here = base_vecs_img[row, col, :]
            base_up = base_vecs_img[row+1, col, :]
            base_diag = base_vecs_img[row+1, col+1, :]
            base_right = base_vecs_img[row, col+1, :]

            val_here = tex_vecs_img[row, col, :]
            val_up = tex_vecs_img[row+1, col, :]
            val_diag = tex_vecs_img[row+1, col+1, :]
            val_right = tex_vecs_img[row, col+1, :]

            trips_list += get_tr_cones_circ(base_here, base_up, base_diag, base_right, val_here, val_up, val_diag, val_right)

    return trips_list


def add_candle_base(outer_radius_mm, candle_radius_mm, n_tris=100, base_bottom_z=0, base_thickness=2):
    rads_per_tri = 2*np.pi / n_tris
    outer_tri_points = [outer_radius_mm * arr([np.sin((i-0.5)*rads_per_tri), np.cos((i-0.5)*rads_per_tri), 0]) for i in range(n_tris)]
    inner_tri_points = [candle_radius_mm * arr([np.sin(i*rads_per_tri), np.cos(i*rads_per_tri), 0]) for i in range(n_tris)]

    trips_list = []
    offset_bottom = arr([0, 0, base_bottom_z])
    offset_top = arr([0, 0, base_thickness])
    for tri_i in range(n_tris):
        o1 = outer_tri_points[tri_i] + offset_bottom
        o2 = outer_tri_points[(tri_i + 1) % n_tris] + offset_bottom
        i1 = inner_tri_points[tri_i] + offset_bottom
        i2 = inner_tri_points[(tri_i + 1) % n_tris] + offset_bottom
        o1o = o1 + offset_top
        o2o = o2 + offset_top
        i1o = i1 + offset_top
        i2o = i2 + offset_top

        # bottom (level 0)
        trips_list += [
            [o2, i1, o1],
            [i1, o2, i2]
        ]

        # top (level offset)
        trips_list += [
            [o1o, i1o, o2o],
            [i2o, o2o, i1o]
        ]

        # outer flat
        trips_list += [
            [o1, o2o, o2],
            [o1o, o2o, o1]
        ]

        # inner flat
        trips_list += [
            [i2, i2o, i1],
            [i1, i2o, i1o]
        ]

    return trips_list


def save_as_stl(outfile_name, trips_list):
    n_trips = len(trips_list)
    data = np.zeros(n_trips, dtype=mesh.Mesh.dtype)
    for i in range(n_trips):
        data['vectors'][i] = arr(trips_list[i])

    my_mesh = mesh.Mesh(data, remove_empty_areas=False)
    my_mesh.save(outfile_name)


def main(is_flat, infile_name, outfile_name, width_mm, radius_mm,
    candle_radius_mm=None, base_height_mm=None, max_depth_mm=2.5,
    min_depth_mm=0.4, min_pixel_width_mm=0.15):

    pic = imageio.imread(infile_name)
    depth_pixels = to_gray(pic)
    depth_pixels = np.flipud(depth_pixels)

    # resize
    circ_mm = 2 * np.pi * radius_mm
    max_cols_px = int(circ_mm / min_pixel_width_mm)
    n_cols = len(depth_pixels[0])
    n_rows = len(depth_pixels)
    if max_cols_px < n_cols:
        row_mult = max_cols_px / n_cols
        rows_px = int(row_mult * n_rows)
        cols_px = max_cols_px
    else:
        cols_px = n_cols
        rows_px = n_rows

    im = Image.fromarray(np.uint8(depth_pixels))
    depth_pixels = np.array(im.resize((cols_px, rows_px)), dtype=float)

    max_val = np.amax(depth_pixels)
    depth_pixels = max_val - depth_pixels  # invert heights
    depth_pixels *= (max_depth_mm - min_depth_mm) / max_val  # scale to given depth range
    depth_pixels += min_depth_mm

    if is_flat:
        trips_list = img_to_planar_mesh(depth_pixels, width_mm)
    else:
        trips_list = img_to_circular_mesh(depth_pixels, radius_mm, max_depth_mm, base_height_mm=base_height_mm)
        if candle_radius_mm is not None:
            trips_list += add_candle_base(radius_mm + max_depth_mm, candle_radius_mm, n_tris=depth_pixels.shape[1])

    save_as_stl(outfile_name, trips_list)


def create_holder(radius_mm, candle_radius_mm=20, base_height_mm=16,
    max_depth_mm=2.5, min_pixel_width_mm=0.15, inset_depth_mm=1, inset_height_mm=1.5, inset_tol_mm=0.1):
    
    inner_radius = radius_mm - inset_depth_mm
    
    n_cells_x = int(2*np.pi*radius_mm / min_pixel_width_mm)
    n_cells_y = int(inset_height_mm / min_pixel_width_mm)
    dummy_depth = (inset_depth_mm - inset_tol_mm) * np.ones((n_cells_y, n_cells_x))
    max_depth_mm += inset_depth_mm

    trips_list = img_to_circular_mesh(dummy_depth, inner_radius, max_depth_mm, base_height_mm=base_height_mm)

    trips_list += add_candle_base(inner_radius + max_depth_mm, candle_radius_mm, n_tris=n_cells_x)
    trips_list += add_candle_base(inner_radius + max_depth_mm, candle_radius_mm=0, n_tris=n_cells_x, base_thickness=1, base_bottom_z=-1)
    
    save_as_stl("holder_inset2.5.stl", trips_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="flat, round, or holder")
    parser.add_argument("infile", nargs='?', help="input image (png, jpeg, etc)")
    parser.add_argument("outfile", nargs='?', help="name for output stl", default=None)
    parser.add_argument("--width", nargs='?', type=float, help="width in mm", default=70)
    parser.add_argument("--radius", nargs='?', type=float, help="radius in mm", default=30)
    parser.add_argument("--candle_radius", nargs='?', type=float, help="radius of candle hole in mm", default=None)
    parser.add_argument("--base_height", nargs='?', type=float, help="height added to base of round lith in mm", default=0)
    parser.add_argument("--max_depth", nargs='?', type=float, help="maximum thickness of lith in mm", default=2.0)
    parser.add_argument("--min_depth", nargs='?', type=float, help="minimum thickness of lith in mm", default=0.7)
    parser.add_argument("--min_pixel_width", nargs='?', type=float, help="minimum width in mm of each pixel in image (reduces image res as needed)", default=0.15)

    args = parser.parse_args()

    assert args.mode in ['flat', 'round', 'holder'], "unsupported mode: " + str(args.mode)
    is_flat = args.mode == 'flat'

    if args.mode == 'holder':
        cr = args.candle_radius or 20  # if none
        bh = args.base_height or 16
        create_holder(args.radius, candle_radius_mm=cr, base_height_mm=bh,
            max_depth_mm=args.max_depth, min_pixel_width_mm=args.min_pixel_width, inset_depth_mm=2.5, inset_height_mm=1.5)
    else:
        assert args.infile is not None, 'This mode requires an input filename'
        if args.outfile is None:
            base_name = ".".join(args.infile.split('.')[:-1])
            if is_flat:
                outfile = '%s-flat.stl' % base_name
            else:
                outfile = '%s-round.stl' % base_name
        else:
            outfile = args.outfile

        main(is_flat=is_flat, infile_name=args.infile, outfile_name=outfile,
            width_mm=args.width, radius_mm=args.radius,
            candle_radius_mm=args.candle_radius, base_height_mm=args.base_height,
            max_depth_mm=args.max_depth, min_depth_mm=args.min_depth,
            min_pixel_width_mm=args.min_pixel_width)
