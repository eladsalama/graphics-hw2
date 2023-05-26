import argparse
from PIL import Image
import numpy as np
import math

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

    # calculating the normalized vector rays from the camera to each pixel on the screen
    rays_from_camera = pixels_positions - camera.position  # tV
    rays_from_camera_norm = rays_from_camera / np.linalg.norm(rays_from_camera, axis=2)[:, :, np.newaxis]  # V

    # finding the intersection of the ray with all surfaces in the scene
    for ray in rays_from_camera_norm.reshape(-1, 3):
        # TODO
        return

    # Dummy result
    image_array = np.zeros((500, 500, 3))

    # Save the output image
    save_image(image_array)


def normalize(v):
    v = np.array(v)
    return v / np.linalg.norm(v)


def get_intersection(P0, V, objects):
    min_t = np.inf
    min_obj = None

    for obj in objects:
        if type(obj) == Sphere:  # based on slide 24 of "Lecture 4 - Ray Casting" presentation.
            O = obj.position  # sphere_center
            L = O - P0

            t_ca = L.dot(V)
            if t_ca < 0:
                return 0, None

            d_squared = L.dot(L) - t_ca ** 2
            r_squared = obj.radius ** 2

            if d_squared > r_squared:
                return 0, None

            t_hc = np.sqrt(r_squared - d_squared)
            t = t_ca - t_hc

        elif type(obj) == InfinitePlane:  # based on slide 26 of "Lecture 4 - Ray Casting" presentation.
            t = -(P0.dot(obj.normal) + obj.offset) / (V.dot(obj.normal))

        else: #type(obj) == Cube
            Nxy=np.array([0,0,1]) #Normal of xy
            Nxz=np.array([0,1,0]) #Normal of xz
            Nyz=np.array([1,0,0]) #Normal of yz

            pmax=obj.position+[obj.scale/2,obj.scale/2,obj.scale/2] # maximal coordinate values in the cube
            pmin=obj.position-[obj.scale/2,obj.scale/2,obj.scale/2] # minimal coordinate values in the cube

            offset1,offset2,offset3,offset4,offset5,offset6=pmin.dot(Nyz),pmin.dot(Nxz),pmin.dot(Nxy),pmax.dot(-Nyz),pmax.dot(-Nxz),pmax.dot(-Nxy)
            
            # xi are the points of intersection,ti are the distances. if  the dot product of V and N is zero then V is parallel to the plane.
            # if V is parallel to the plane i we set xi and xi+3 to be points with coordinates bigger than pmax and  we set ti and ti+3 to be -1.
            #else we calculate ti ti+3 xi and xi+3 as if its an infinite plane 
            if ((V.dot(Nyz))==0):
                t1,t4,x1,x4=-1,-1,pmax+np.array([1,1,1]),pmax+np.array([1,1,1])
            else:
                 t1=-(P0.dot(Nyz) + offset1)/(V.dot(Nyz));t4=-(P0.dot(-Nyz) + offset4)/(V.dot(-Nyz))
                 x1,x4=P0+t1*V,P0+t4*V
            if ((V.dot(Nxz))==0):
                t2,t5,x2,x5=-1,-1,pmax+np.array([1,1,1]),pmax+np.array([1,1,1]) 
            else:
                t2=-(P0.dot(Nxz) + offset2)/(V.dot(Nxz)); t5=-(P0.dot(-Nxz) + offset5)/(V.dot(-Nxz))
                x2,x5=P0+t2*V,P0+t5*V
            if ((V.dot(Nxy))==0):
                t3,t6,x3,x6=-1,-1,pmax+np.array([1,1,1]),pmax+np.array([1,1,1])
            else:
                t3=-(P0.dot(Nxy) + offset3)/(V.dot(Nxy));t6=-(P0.dot(-Nxy) + offset6)/(V.dot(-Nxy))
                x3,x6=P0+t3*V,P0+t6*V              
            
            # if the point of intersection has one coordinate higher than max or lower than min, 
            # then the intersetion with the plane is out of the cube. 
            # note that if the plane is parallel to xy then there is no need to check the z coordinate and so on.
            if (x1[1]>pmax[1] or x1[1]<pmin[1] or x1[2]>pmax[2] or x1[2]<pmin[2]):
                t1=-1
            if (x2[0]>pmax[0] or x2[0]<pmin[0] or x2[2]>pmax[2] or x2[2]<pmin[2]):
                t2=-1
            if (x3[0]>pmax[0] or x3[0]<pmin[0] or x3[1]>pmax[1] or x3[1]<pmin[1]):
                t3=-1
            if (x4[1]>pmax[1] or x4[1]<pmin[1] or x4[2]>pmax[2] or x4[2]<pmin[2]):
                t4=-1
            if (x5[0]>pmax[0] or x5[0]<pmin[0] or x5[2]>pmax[2] or x5[2]<pmin[2]):
                t5=-1
            if (x6[0]>pmax[0] or x6[0]<pmin[0] or x6[1]>pmax[1] or x6[1]<pmin[1]):
                t6=-1
            T=np.array([t1,t2,t3,t4,t5,t6])
            T=T[np.sort(T >= 0)]

            if (T.shape[1]==0):#asserts that there is an intersection  
                t=math.inf
            else:#the first t in T is the distance from the first face of the cube to be hit
                t=T[0]
                
        if t < min_t:
            min_t = t
            min_obj = obj

    return min_t, min_obj

def calc_specular_reflection(V, N, L, obj, light,material):
    Ks =  material.specular_color
    n = material.shininess  
    Il = light.specular_intensity*light.color  

    R = (2*L).dot(N) - L  # R = (2LN)N - L

    #Id =  Is = Ks*Il*(V.dot(R))**n ?
    Is = np.multiply(Ks, np.multiply(Il, (V.dot(R))**n))
    return Is

def calc_diffuse_reflection(V, N, L, obj, light,material):
    Kd = material.diffuse_color
    Il =light.color    

    #Id = Kd*(N.dot(L))*Il?
    Id = np.multiply(Kd, np.multiply(Il, (V.dot(L))))

    return Id


if __name__ == '__main__':
    main()
