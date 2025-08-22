import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import matplotlib.colors as mcolors


def plot_2d_projection(pixels):

    # 2. Extract the pixel data and info
    pixel_list = pixels["pixels"]
    npix = pixels["info"]["npix"]
    nside = pixels["info"]["nside"]

    # 3. Create the Matplotlib figure and projection
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection="mollweide")

    # 4. Create polygons, splitting them if they cross the 180-degree meridian
    # The vertex data is already in the correct [lat, lon] format
    patches_list = []
    # Create a simple data map (pixel index) for coloring
    color_data = np.arange(npix)
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=color_data.min(), vmax=color_data.max())

    for i, p_data in enumerate(pixel_list):
        # p_data structure: [ [center_lat, center_lon], [ [v1_lat, v1_lon], ... ] ]
        pixel_verts = np.array(p_data[1])  # Vertices are the second element
        pixel_lons = pixel_verts[:, 1]  # Longitude is the second coordinate

        # Check for wrapping polygons
        if np.max(pixel_lons) - np.min(pixel_lons) > np.pi:
            # Split the polygon into two parts

            # Part 1: Shift negative longitudes to the right
            patch1_verts = pixel_verts.copy()
            # Note: We work with longitude/latitude here, so index is 1
            patch1_verts[:, 1] = np.where(
                patch1_verts[:, 1] < 0,
                patch1_verts[:, 1] + 2 * np.pi,
                patch1_verts[:, 1],
            )
            # For patches, matplotlib expects (lon, lat)
            polygon1 = patches.Polygon(
                patch1_verts[:, ::-1], closed=True, facecolor=cmap(norm(color_data[i]))
            )

            # Part 2: Shift positive longitudes to the left
            patch2_verts = pixel_verts.copy()
            patch2_verts[:, 1] = np.where(
                patch2_verts[:, 1] > 0,
                patch2_verts[:, 1] - 2 * np.pi,
                patch2_verts[:, 1],
            )
            polygon2 = patches.Polygon(
                patch2_verts[:, ::-1], closed=True, facecolor=cmap(norm(color_data[i]))
            )

            patches_list.extend([polygon1, polygon2])
        else:
            # Normal polygon
            # Matplotlib's Polygon expects (x, y), which corresponds to (lon, lat)
            # Our data is (lat, lon), so we reverse the order with [:, ::-1]
            polygon = patches.Polygon(
                pixel_verts[:, ::-1], closed=True, facecolor=cmap(norm(color_data[i]))
            )
            patches_list.append(polygon)

    # 5. Add the collection of patches to the plot for efficient rendering
    collection = PatchCollection(patches_list, match_original=True)
    ax.add_collection(collection)

    # 6. Customize and show the plot
    ax.set_title(f"HEALPix Grid from JSON (NSIDE={nside})")
    ax.grid(True)
    ax.autoscale_view()

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, pad=0.1, aspect=40, ax=ax)
    cbar.set_label("Pixel Index")

    plt.show()


def plot_3d_scatter(pixels):

    # 2. Extract the pixel center coordinates
    # The structure is [[center_coords], [vertex_coords]], so we take the first element
    pixel_centers_spherical = np.array([p_data[0] for p_data in pixels["pixels"]])
    nside = pixels["info"]["nside"]

    # Separate latitude and longitude
    # Our data is stored as [latitude, longitude]
    lat = pixel_centers_spherical[:, 0]
    lon = pixel_centers_spherical[:, 1]

    # 3. Convert spherical coordinates back to 3D Cartesian (x, y, z)
    # x = cos(lat) * cos(lon)
    # y = cos(lat) * sin(lon)
    # z = sin(lat)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    # 4. Create the 3D plot
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")

    # Use pixel index for coloring
    color_data = np.arange(pixels["info"]["npix"])

    # Create the scatter plot
    scatter = ax.scatter(x, y, z, c=color_data, cmap="viridis", s=5)  # s is point size

    # 5. Customize the plot
    ax.set_title(f"HEALPix 3D Point Cloud from JSON (NSIDE={nside})")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Set aspect ratio to be equal, creating a sphere
    # This is a bit tricky in Matplotlib 3D, but this is the standard approach
    ax.set_box_aspect([1, 1, 1])  # For Matplotlib 3.1+

    # Add a color bar
    cbar = fig.colorbar(scatter, shrink=0.6, aspect=20)
    cbar.set_label("Pixel Index")

    plt.show()


def plot_3d_vectors(pixels):

    nside = pixels["info"]["nside"]

    # 2. Extract pixel center coordinates (latitude, longitude)
    pixel_centers_spherical = np.array([p_data[0] for p_data in pixels["pixels"]])
    lat = pixel_centers_spherical[:, 0]
    lon = pixel_centers_spherical[:, 1]

    # 3. Convert spherical coordinates to 3D Cartesian (x, y, z) for vector endpoints
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    # 4. Set up the 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # 5. Prepare data for the quiver plot
    # All vectors start at the origin (0, 0, 0)
    origin = np.zeros_like(x)

    # Prepare colors for the vectors based on pixel index
    npix = pixels["info"]["npix"]
    color_indices = np.arange(npix)
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=color_indices.min(), vmax=color_indices.max())

    # Use the quiver function to plot the vectors
    ax.quiver(
        origin,
        origin,
        origin,  # Start X, Y, Z
        x,
        y,
        z,  # End X, Y, Z (as vector components)
        colors=cmap(norm(color_indices)),  # Set color for each arrow
        length=1.0,  # Length of the arrow shaft
        arrow_length_ratio=0.1,  # Size of the arrowhead
        normalize=False,
    )

    # 6. Customize the plot
    ax.set_title(f"HEALPix 3D Vectors from JSON (NSIDE={nside})")
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")

    # Set plot limits to keep the sphere centered
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])

    # Add a color bar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # You need to set a dummy array for the color bar to work
    cbar = fig.colorbar(sm, shrink=0.6, aspect=20, ax=ax)
    cbar.set_label("Pixel Index")

    plt.show()
