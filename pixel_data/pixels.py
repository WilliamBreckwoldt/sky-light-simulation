import json
import os
import numpy as np


def load_healpix_json(nside, data_dir="."):
    """
    Loads pre-calculated HEALPix data from a JSON file and plots it
    on a Mollweide projection using Matplotlib.

    Args:
        nside (int): The nside resolution of the map to plot. Must be a power of 2.
        data_dir (str): The directory where the JSON files are stored.
    """
    # 1. Construct the file path and load the data
    file_path = os.path.join(data_dir, f"healpix_nside_{nside}.json")
    if not os.path.exists(file_path):
        print(f"Error: Data file not found at '{file_path}'")
        return

    # print(f"Loading data from '{file_path}'...")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_healpix_json(nside_max=128):
    """
    Generates and saves JSON files containing pre-calculated HEALPix grid data.

    For each valid nside value up to nside_max, this function calculates the
    pixel center and vertex coordinates in a Matplotlib-compatible format
    (latitude/longitude) and saves them to a file named 'healpix_nside_{nside}.json'.

    Args:
        nside_max (int): The maximum nside value to process. Must be a power of 2.
    """
    # import here so users without healpy can use other functions in this file
    import healpy as hp  # pylint: disable=C0415

    # Generate a list of valid nside values (powers of 2)
    nside_values = [2**i for i in range(int(np.log2(nside_max)) + 1)]

    for nside in nside_values:
        print(f"Processing nside = {nside}...")

        # 1. Get the number of pixels
        npix = hp.nside2npix(nside)

        # 2. Calculate pixel center coordinates
        pix_indices = np.arange(npix)
        theta_pix, phi_pix = hp.pix2ang(nside, pix_indices)

        # 3. Get pixel boundary vertex coordinates (shape: npix, 3, 4)
        vertices_3d = hp.boundaries(nside, pix_indices)
        # Transpose to (npix, 4, 3) for easier processing
        vertices_3d_swapped = np.transpose(vertices_3d, (0, 2, 1))
        # Flatten for batch conversion
        vertices_flat_3d = vertices_3d_swapped.reshape(-1, 3)
        theta_vert, phi_vert = hp.vec2ang(vertices_flat_3d)

        # 4. Define a function to convert coordinates to Matplotlib format
        def convert_coords(theta, phi):
            """Converts HEALPix (theta, phi) to Matplotlib (lat, lon)."""
            lon = phi.copy()
            # Wrap longitude from [0, 2*pi] to [-pi, pi]
            lon[lon > np.pi] -= 2 * np.pi
            # Convert polar angle theta [0, pi] to latitude [-pi/2, pi/2]
            lat = np.pi / 2 - theta
            return lat, lon

        # 5. Convert all coordinates
        lat_pix, lon_pix = convert_coords(theta_pix, phi_pix)
        lat_vert, lon_vert = convert_coords(theta_vert, phi_vert)

        # 6. Reshape vertex data back to be grouped by pixel
        # Shape: (npix, 4, 2) -> (pixel, vertex, lat_lon)
        vertices_2d = np.dstack([lat_vert, lon_vert]).reshape(npix, 4, 2)

        # 7. Assemble the final data structure
        pixel_data = []
        for i in range(npix):
            pixel_center = [lat_pix[i], lon_pix[i]]
            pixel_vertices = vertices_2d[i].tolist()
            pixel_data.append([pixel_center, pixel_vertices])

        output_json = {
            "info": {
                "nside": nside,
                "npix": npix,
                "data_info": "pixel is a list of information, each element is described by data_shape. Each pixel and vert is described in (theta [-pi/2, pi/2], phi [-pi, pi]) coordinates. The four verts associated with each pixel describe the vertices of the pixel's region.",
                "data_shape": [
                    ["pixel_theta", "pixel_phi"],
                    [
                        ["vert1_theta", "vert1_phi"],
                        ["vert2_theta", "vert2_phi"],
                        ["vert3_theta", "vert3_phi"],
                        ["vert4_theta", "vert4_phi"],
                    ],
                ],
            },
            "pixels": pixel_data,
        }

        # 8. Write the data to a JSON file
        # file_path = os.path.join(output_dir, f"healpix_nside_{nside}.json")
        file_name = f"healpix_nside_{nside}.json"
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(
                output_json, f
            )  # Use indent for pretty-printing: json.dump(output_json, f, indent=2)

    print("\nProcessing complete.")
