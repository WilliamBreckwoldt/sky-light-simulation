import json
import os
import numpy as np

# Define the range of a 16-bit signed integer for scaling.
# A 16-bit integer can store values from -32768 to 32767. We use 32767 for symmetry.
INT16_MAX = 32767.0


def generate_all_healpix_data(nside_max):
    nside_values = [2**i for i in range(int(np.log2(nside_max)) + 1)]

    for nside in nside_values:
        generate_and_save_healpix_data(nside)


def generate_and_save_healpix_data(nside, data_dir=".", force_regenerate=False):
    """
    Generates HEALPix grid data, saves the numerical data to a compact .npy
    binary file using 16-bit integer quantization, and saves the corresponding
    metadata to a .json file.

    Args:
        nside (int): The nside resolution of the map. Must be a power of 2.
        data_dir (str): The directory where the files will be stored.
        force_regenerate (bool): If True, will regenerate files even if they
                                 already exist.
    """
    # 1. Define file paths and check for existence
    json_path = os.path.join(data_dir, f"healpix_nside_{nside}.json")
    npy_path = os.path.join(data_dir, f"healpix_nside_{nside}.npy")

    if not force_regenerate and os.path.exists(json_path) and os.path.exists(npy_path):
        print(f"Files for nside={nside} already exist. Skipping generation.")
        return

    print(f"Processing nside = {nside}...")

    # 2. Perform HEALPix calculations (similar to the original code)
    try:
        import healpy as hp
    except ImportError:
        print("Error: healpy is required to generate new data. Please install it.")
        return

    npix = hp.nside2npix(nside)
    pix_indices = np.arange(npix)
    theta_pix, phi_pix = hp.pix2ang(nside, pix_indices)
    vertices_3d = hp.boundaries(nside, pix_indices)
    vertices_3d_swapped = np.transpose(vertices_3d, (0, 2, 1))
    vertices_flat_3d = vertices_3d_swapped.reshape(-1, 3)
    theta_vert, phi_vert = hp.vec2ang(vertices_flat_3d)

    def convert_coords(theta, phi):
        lon = phi.copy()
        lon[lon > np.pi] -= 2 * np.pi
        lat = np.pi / 2 - theta
        return lat, lon

    lat_pix, lon_pix = convert_coords(theta_pix, phi_pix)
    lat_vert, lon_vert = convert_coords(theta_vert, phi_vert)

    # 3. Assemble all float data into a single NumPy array
    # Structure: [center_lat, center_lon, v1_lat, v1_lon, v2_lat, v2_lon, ...]
    centers_array = np.dstack([lat_pix, lon_pix]).reshape(npix, 2)
    vertices_array = np.dstack([lat_vert, lon_vert]).reshape(npix, 8)
    float_array = np.hstack([centers_array, vertices_array]).astype(np.float32)

    # 4. Quantize the data: Convert the float array to a 16-bit integer array
    # We map the range [-2*pi, 2*pi] to the int16 range [-32767, 32767]
    scaling_factor = INT16_MAX / (2 * np.pi)
    quantized_array = (float_array * scaling_factor).astype(np.int16)

    # 5. Save the quantized array to the .npy file
    np.save(npy_path, quantized_array)
    print(f"Successfully saved binary data to '{npy_path}'")

    # 6. Create and save the metadata JSON
    output_json = {
        "info": {
            "nside": nside,
            "npix": npix,
            "binary_file": os.path.basename(npy_path),
            "data_type": "int16",
            "scaling_factor": scaling_factor,
            "data_info": "Pixel data is stored in the binary file. To decode, load the int16 data and divide by the scaling_factor. Each row corresponds to one pixel and contains 10 values.",
            "data_shape": [
                ["pixel_theta", "pixel_phi"],
                [
                    ["vert1_theta", "vert1_phi"],
                    ["vert2_theta", "vert2_phi"],
                    ["vert3_theta", "vert3_phi"],
                    ["vert4_theta", "vert4_phi"],
                ],
            ],
        }
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)
    print(f"Successfully saved metadata to '{json_path}'")


def load_healpix_data(nside, data_dir="."):
    """
    Loads HEALPix metadata from a JSON file and the corresponding binary
    pixel data from a .npy file, reconstructing it into the original
    dictionary format.

    Args:
        nside (int): The nside resolution of the map to load.
        data_dir (str): The directory where the data files are stored.

    Returns:
        dict: A dictionary with the same structure as the original JSON,
              containing the 'info' and reconstructed 'pixels' data.
              Returns None if files are not found.
    """
    # 1. Load the metadata from the JSON file
    json_path = os.path.join(data_dir, f"healpix_nside_{nside}.json")
    if not os.path.exists(json_path):
        print(f"Error: Metadata file not found at '{json_path}'")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    info = metadata["info"]

    # 2. Load the binary data from the .npy file
    npy_path = os.path.join(data_dir, info["binary_file"])
    if not os.path.exists(npy_path):
        print(f"Error: Binary data file not found at '{npy_path}'")
        return None

    # 3. De-quantize: Convert the integer data back to floats
    scaling_factor = info["scaling_factor"]
    float_array = np.load(npy_path).astype(np.float32) / scaling_factor

    # 4. Reconstruct the original nested list structure
    pixel_data = []
    npix = info["npix"]

    # Separate the centers and vertices using efficient NumPy slicing
    centers_array = float_array[:, 0:2]
    vertices_array = float_array[:, 2:10].reshape(npix, 4, 2)

    # Convert the entire NumPy arrays to Python lists in one highly optimized call
    centers_list = centers_array.tolist()
    vertices_list = vertices_array.tolist()

    # Use the built-in zip() function to pair each center with its corresponding
    # vertices. A final list comprehension assembles the final structure.
    # This is significantly faster than appending one by one in a loop.
    pixel_data = [list(pair) for pair in zip(centers_list, vertices_list)]

    # 5. Assemble and return the final dictionary
    return {"info": info, "pixels": pixel_data}
