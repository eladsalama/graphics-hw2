import argparse
from PIL import Image
import numpy as np
import math

import scene_settings
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
    #parser = argparse.ArgumentParser(description='Python Ray Tracer')
    #parser.add_argument('scene_file', type=str, help='Path to the scene file')
    #parser.add_argument('output_image', type=str, help='Name of the output image file')
    #parser.add_argument('--width', type=int, default=50, help='Image width')
    #parser.add_argument('--height', type=int, default=50, help='Image height')
    #args = parser.parse_args()

    # Parse the scene file
    #camera, scene_settings, objects = parse_scene_file(args.scene_file)
    camera, scene_settings, objects = parse_scene_file(
        r"C:\Users\Elad\Documents\Python Projects\basics of graphics\hw2\scenes\pool.txt")

    # TODO: Implement the ray tracer
    #width, height = args[2], args[3]
    width, height = 50, 50
    image_array = np.zeros((width, height, 3))

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

    # calculating the normalized vector rays from the camera to each pixel on the screen
    rays_from_camera = pixels_positions - camera.position  # tV
    rays_from_camera_norm = rays_from_camera / np.linalg.norm(rays_from_camera, axis=2)[:, :, np.newaxis]  # V

    # finding the intersection of the ray with all surfaces in the scene
    P0 = np.array(camera.position)
    lights = [item for item in objects if isinstance(item, Light)]
    surfaces = [item for item in objects if isinstance(item, (InfinitePlane, Sphere, Cube))]
    materials = [item for item in objects if isinstance(item, Material)]

    for i in range(rays_from_camera_norm.shape[0]):
        for j in range(rays_from_camera_norm.shape[1]):
            ray = rays_from_camera_norm[i][j]
            color = trace_ray(P0, ray, surfaces, lights, materials,
                              scene_settings.background_color, None, scene_settings.max_recursions)
            image_array[i][j] = color

    print(image_array[0][0])
    # Save the output image
    save_image(image_array)


def normalize(v):
    v = np.array(v)

    if np.linalg.norm(v) != 0:
        return v / np.linalg.norm(v)
    return v


def get_intersection(P0, V, objects, whitelist):
    """ returns the closest intersecting object, the intersecting point (P0+tV),
        and the outwards normal in the intersection. """
    min_obj = None
    min_t = np.inf
    min_outwards_normal = [0, 0, 0]

    for obj in objects:
        if obj in whitelist:
            continue

        if type(obj) == Sphere:  # based on slide 24 of "Lecture 4 - Ray Casting" presentation.
            O = np.array(obj.position)  # sphere_center
            L = O - P0

            t_ca = L.dot(V)
            if t_ca < 0:
                continue

            d_squared = L.dot(L) - t_ca ** 2
            r_squared = obj.radius ** 2

            if d_squared > r_squared:
                continue

            t_hc = np.sqrt(r_squared - d_squared)
            t = t_ca - t_hc
            outwards_normal = (P0 + t * V) - O
            # print(f"sphere. t={t}")

        elif type(obj) == InfinitePlane:  # based on slide 26 of "Lecture 4 - Ray Casting" presentation.
            N = normalize(obj.normal)
            t = -(P0.dot(N) + obj.offset) / (V.dot(N))
            outwards_normal = get_outwards_normal(N, V)
            # print(f"plane. t={t}")

        else:  # type(obj) == Cube
            Nxy = np.array([0, 0, 1])  # Normal of xy
            Nxz = np.array([0, 1, 0])  # Normal of xz
            Nyz = np.array([1, 0, 0])  # Normal of yz

            pmax = obj.position + [obj.scale / 2, obj.scale / 2, obj.scale / 2]  # maximal coordinate values in the cube
            pmin = obj.position - [obj.scale / 2, obj.scale / 2, obj.scale / 2]  # minimal coordinate values in the cube

            offset1, offset2, offset3, offset4, offset5, offset6 = pmin.dot(Nyz), pmin.dot(Nxz), pmin.dot(
                Nxy), pmax.dot(-Nyz), pmax.dot(-Nxz), pmax.dot(-Nxy)

            # xi are the points of intersection, ti are the distances.
            # if the dot product of V and N is zero then V is parallel to the plane.
            # if V is parallel to the plane i we set xi and xi+3 to be points with coordinates bigger than pmax,
            # and we set ti and ti+3 to be -1.
            # else, we calculate ti, ti+3 xi and xi+3 as if it is an infinite plane
            if V.dot(Nyz) == 0:
                t1, t4, x1, x4 = -1, -1, pmax + np.array([1, 1, 1]), pmax + np.array([1, 1, 1])
            else:
                t1 = -(P0.dot(Nyz) + offset1) / (V.dot(Nyz))
                t4 = -(P0.dot(-Nyz) + offset4) / (V.dot(-Nyz))
                x1, x4 = P0 + t1 * V, P0 + t4 * V

            if V.dot(Nxz) == 0:
                t2, t5, x2, x5 = -1, -1, pmax + np.array([1, 1, 1]), pmax + np.array([1, 1, 1])
            else:
                t2 = -(P0.dot(Nxz) + offset2) / (V.dot(Nxz))
                t5 = -(P0.dot(-Nxz) + offset5) / (V.dot(-Nxz))
                x2, x5 = P0 + t2 * V, P0 + t5 * V

            if V.dot(Nxy) == 0:
                t3, t6, x3, x6 = -1, -1, pmax + np.array([1, 1, 1]), pmax + np.array([1, 1, 1])
            else:
                t3 = -(P0.dot(Nxy) + offset3) / (V.dot(Nxy))
                t6 = -(P0.dot(-Nxy) + offset6) / (V.dot(-Nxy))
                x3, x6 = P0 + t3 * V, P0 + t6 * V

            # if the point of intersection has one coordinate higher than max or lower than min,
            # then the intersection with the plane is out of the cube.
            # note that if the plane is parallel to xy then there is no need to check the z coordinate and so on.
            if x1[1] > pmax[1] or x1[1] < pmin[1] or x1[2] > pmax[2] or x1[2] < pmin[2]:
                t1 = -1
            if x2[0] > pmax[0] or x2[0] < pmin[0] or x2[2] > pmax[2] or x2[2] < pmin[2]:
                t2 = -1
            if x3[0] > pmax[0] or x3[0] < pmin[0] or x3[1] > pmax[1] or x3[1] < pmin[1]:
                t3 = -1
            if x4[1] > pmax[1] or x4[1] < pmin[1] or x4[2] > pmax[2] or x4[2] < pmin[2]:
                t4 = -1
            if x5[0] > pmax[0] or x5[0] < pmin[0] or x5[2] > pmax[2] or x5[2] < pmin[2]:
                t5 = -1
            if x6[0] > pmax[0] or x6[0] < pmin[0] or x6[1] > pmax[1] or x6[1] < pmin[1]:
                t6 = -1
            T = np.array([t1, t2, t3, t4, t5, t6])
            T = T[np.sort(T >= 0)]

            if T.shape[1] == 0:  # asserts that there is an intersection
                t = math.inf
            else:  # the first t in T is the distance from the first face of the cube to be hit
                t = T[0]
                if t == t1:
                    outwards_normal = Nyz
                elif t == t2:
                    outwards_normal = Nxz
                elif t == t3:
                    outwards_normal = Nxy
                elif t == t4:
                    outwards_normal = -Nyz
                elif t == t5:
                    outwards_normal = -Nxz
                elif t == t6:
                    outwards_normal = -Nxy

                # print(f"cube. t={t}")

        if t < min_t:
            min_t = t
            min_obj = obj
            min_outwards_normal = outwards_normal

    return min_obj, min_t, normalize(min_outwards_normal)


def get_outwards_normal(surface_normal, incoming_ray):
    # If the angle between the surface normal and the incoming ray is greater than 90 degrees,
    # the surface normal is pointing inwards and not outwards.
    angle = np.arccos(np.dot(surface_normal, -incoming_ray))
    if angle > np.pi / 2:
        outwards_normal = -surface_normal
    else:
        outwards_normal = surface_normal

    return outwards_normal


def calc_specular_reflection(V, N, L, light, light_color, material):
    """ send normalized vectors V, N, L """
    Ks = np.array(material.specular_color)
    n = material.shininess
    Il = np.array(light.specular_intensity) * light_color

    R = normalize(L - 2 * np.dot(L, N) * N)  # reflection direction

    Is = Ks * Il * (V.dot(R))**n  # based on slide 45 of "Lecture 4 - Ray Casting" presentation.
    return Is


def calc_diffuse_reflection(N, L, light_color, material):
    """ send normalized vectors N, L """
    Kd = np.array(material.diffuse_color)
    Il = light_color

    Id = Kd * (N.dot(L)) * Il  # based on slide 42 of "Lecture 4 - Ray Casting" presentation.
    return Id


def trace_ray(P0, V, surfaces, lights, materials, bg_color, prev_obj, recursion_depth):
    if recursion_depth == 0:
        return 0

    surf, t, N = get_intersection(P0, V, surfaces, [prev_obj])

    if t == np.inf:
        # print("no intersection")
        return bg_color

    mat = materials[surf.material_index - 1]

    phong_color = 0

    for light in lights:
        L = normalize(light.position - (P0 + t * V))
        S = 1

        angle = np.arccos(np.dot(N, L))
        # print(f"angle of light and normal: {angle}")
        if angle > np.pi / 2:  # when the light doesn't hit the surface directly
            # print("the light doesn't hit the surface directly")
            S = 0

        objects_inbetween = []
        materials_inbetween = []
        i = 0
        while S == 1:
            obj_inbetween = get_intersection(np.array(light.position), -L, surfaces, objects_inbetween)[0]
            mat_inbetween = materials[obj_inbetween.material_index - 1]

            if obj_inbetween == surf:
                # print("no more objects inbetween")
                break

            if mat_inbetween.transparency == 0:
                S = 0

            # print("appended object to inbetween", obj_inbetween)
            objects_inbetween.append(obj_inbetween)
            materials_inbetween.append(mat_inbetween)

            i += 1

        if S == 1:  # if the light hits the point
            # print("the light hits the surface directly")
            distorted_light = np.array(light.color)
            for m in materials_inbetween:  # to is a shortcut for transparent object
                # print(f"distorting the light with material inbetween: {m}")
                transparency = m.transparency
                diffuse = m.diffuse_color
                specular = m.specular_color
                reflection = m.reflection_color

                distorted_light = distorted_light * transparency + (diffuse + specular) * (
                            1 - transparency) + reflection

            diffuse_reflection = calc_diffuse_reflection(N, L, distorted_light, mat)
            specular_reflection = calc_specular_reflection(V, N, L, light, distorted_light, mat)
            phong_color += diffuse_reflection + specular_reflection

    if mat.transparency != 0:  # if transparent
        # print("object transparent")
        P0_next = P0  # coordinates of next face the ray hits in the object
        if type(surf) == Cube or type(surf) == Sphere:
            epsilon = 0.005
            t_next = get_intersection(P0 + (t + epsilon) * V, V, surfaces, [])[1]
            # print(t_next)
            P0_next = P0 + t_next * V

        behind_color = trace_ray(P0_next, V, surfaces, lights, materials, bg_color, surf, recursion_depth)
        transparency = mat.transparency
        diffuse = mat.diffuse_color
        specular = mat.specular_color
        reflection = mat.reflection_color

        color = behind_color * transparency + (diffuse + specular) * (1 - transparency) + reflection
        return color

    else:
        # reflected ray:
        P0 = P0 + t * V
        V = normalize(V - 2 * np.dot(V, N) * N)
        # print("calculating reflected color")
        reflected_color = trace_ray(P0, V, surfaces, lights, materials, bg_color, surf, recursion_depth - 1)

        color = reflected_color * np.array(mat.reflection_color) + phong_color

    # print(f"final color {color}")
    return color


if __name__ == '__main__':
    main()

# TODO: find out how to compute PHONG (what are all these variables?!?)
# TODO: why doesnt transparent objects use PHONE formula
# TODO: fix code:
# TODO: things that worry me:
# TODO: 1. לגבי חפצים שקופים, האם הפתרון עם אפסילון עובד. כלומר האם צריך להמשיך כאשר הנקודה ההתחלתית היא הפאה שממנה הקשת יוצאת
# TODO: 2. בדקנו רק לגבי מישור. צריך לבדוק גם כדור וגם קובייה
# TODO: soft shadows