import argparse
from PIL import Image
import numpy as np

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import InfinitePlane
from surfaces.sphere import Sphere


def parse_scene_file(file_path):
    objects = []
    camera = None
    scene_settings = None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                objects.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]))
                objects.append(sphere)
            elif obj_type == "pln":
                plane = InfinitePlane(params[:3], params[3], int(params[4]))
                objects.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]))
                objects.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                objects.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, objects


def save_image(image_array):
    image = Image.fromarray(np.uint8(image_array))

    # Save the image to a file
    image.save("scenes/Spheres.png")


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, help='Path to the scene file')
    parser.add_argument('output_image', type=str, help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, objects = parse_scene_file(args.scene_file)

    # TODO: Implement the ray tracer
    width, height = args[2], args[3]
    aspect_ratio = width / height
    screen_height = camera.screen_width / aspect_ratio
    pixel_width = camera.screen_width / width
    pixel_height = screen_height / height

    # basic vectors
    towards = normalize(np.array(camera.look_at) - np.array(camera.position))  # maybe need to multiply by -1
    up = normalize(camera.up_vector)
    right = normalize(np.cross(towards, up))
    up_perp = normalize(np.cross(right, towards))

    Pc = camera.position + camera.screen_distance * towards  # screen center

    # calculating the pixels positions based on slide 19 of "Lecture 4 - Ray Casting" presentation:
    j_row = (np.arange(width) - np.floor(width / 2)) * pixel_width
    j_matrix = np.tile(j_row, (height, 1))
    j_matrix_3d = np.repeat(j_matrix[:, :, np.newaxis], 3, axis=2)

    i_column = (np.arange(height) - np.floor(height / 2)) * pixel_height
    i_matrix = np.tile(i_column, (width, 1)).T
    i_matrix_3d = np.repeat(i_matrix[:, :, np.newaxis], 3, axis=2)

    pixels_positions = Pc + j_matrix_3d * right - i_matrix_3d * up_perp
    rays_from_camera = pixels_positions - camera.position



    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v)

if __name__ == '__main__':
    main()
