import os
import cv2
import numpy as np
import pandas as pd
from scipy.misc import imresize
import scipy.io


dataset_path = '/data/Research/DiscoGAN/datasets'
celebA_path = os.path.join(dataset_path, 'celebA')
handbag_path = os.path.join(dataset_path, 'edges2handbags')
shoe_path = os.path.join(dataset_path, 'edges2shoes')
facescrub_path = os.path.join(dataset_path, 'facescrub')
chair_path = os.path.join(dataset_path, 'rendered_chairs')
face_3d_path = os.path.join(dataset_path, 'PublicMM1', '05_renderings')
face_real_path = os.path.join(dataset_path, 'real_face')
car_path = os.path.join(dataset_path, 'data', 'cars')

def shuffle_data(da, db): # change range(len()) to list(range(len)) !!
    a_idx = list(range(len(da)))
    np.random.shuffle( a_idx )

    b_idx = list(range(len(db)))
    np.random.shuffle(b_idx)

    shuffled_da = np.array(da)[ np.array(a_idx) ]
    shuffled_db = np.array(db)[ np.array(b_idx) ]

    return shuffled_da, shuffled_db

def read_images( filenames, domain=None, image_size=64):

    images = []
    for fn in filenames:
        image = cv2.imread(fn)  # fn should the string of name of the image, for example: '1.A.jpg'
        if image is None:
            continue

        if domain == 'A':
            kernel = np.ones((3,3), np.uint8)
            image = image[:, :256, :]
            image = 255. - image
            image = cv2.dilate( image, kernel, iterations=1 )
            image = 255. - image
        elif domain == 'B':
            image = image[:, 256:, :]

        image = cv2.resize(image, (image_size,image_size))
        image = image.astype(np.float32) / 255.
        image = image.transpose(2,0,1)
        images.append( image )

    images = np.stack( images )
    return images

def read_attr_file( attr_path, image_dir ):
    f = open( attr_path )
    lines = f.readlines()
    lines = map(lambda line: line.strip(), lines)
    columns = ['image_path'] + lines[1].split()
    lines = lines[2:]

    items = map(lambda line: line.split(), lines)
    df = pd.DataFrame( items, columns=columns )
    df['image_path'] = df['image_path'].map( lambda x: os.path.join( image_dir, x ) )

    return df

def get_celebA_files(style_A, style_B, constraint, constraint_type, test=False, n_test=200):
    attr_file = os.path.join( celebA_path, 'list_attr_celeba.txt' )
    image_dir = os.path.join( celebA_path, 'img_align_celeba' )
    image_data = read_attr_file( attr_file, image_dir )

    if constraint:
        image_data = image_data[ image_data[constraint] == constraint_type]

    style_A_data = image_data[ image_data[style_A] == '1']['image_path'].values
    if style_B:
        style_B_data = image_data[ image_data[style_B] == '1']['image_path'].values
    else:
        style_B_data = image_data[ image_data[style_A] == '-1']['image_path'].values

    if test == False:
        return style_A_data[:-n_test], style_B_data[:-n_test]
    if test == True:
        return style_A_data[-n_test:], style_B_data[-n_test:]


def get_spv_edge2photo_files(my_idx, item='edges2shoes',num_spv=500):
    if item == 'edges2handbags':
        item_path = handbag_path
    elif item == 'edges2shoes':
        item_path = shoe_path
    item_path = os.path.join( item_path, 'train' )

    image_paths = map(lambda x: os.path.join( item_path, x ), os.listdir( item_path ))
    return np.array(image_paths)[np.array(my_idx[:num_spv])]


def get_edge2photo_files(item='edges2shoes', test=False, use_spv_train=False,num_spv=500):
    if item == 'edges2handbags':
        item_path = handbag_path
    elif item == 'edges2shoes':
        item_path = shoe_path
    if test == True:
        item_path = os.path.join( item_path, 'val' )
    else:
        item_path = os.path.join( item_path, 'train' )
    image_paths = map(lambda x: os.path.join( item_path, x ), os.listdir( item_path ))
    n_images = len( image_paths )
    my_idx = list(range(n_images))
    np.random.shuffle(my_idx)
    if use_spv_train == True:
        return np.array(image_paths)[np.array(my_idx[num_spv:])], my_idx
    if test == True:
        return [image_paths, image_paths]
    else:
        n_images = len( image_paths )
        return [image_paths[:n_images/2], image_paths[n_images/2:]]


def get_facescrub_files(test=False, n_test=200):
    actor_path = os.path.join(facescrub_path, 'actors', 'face' )
    actress_path = os.path.join( facescrub_path, 'actresses', 'face' )

    actor_files = map(lambda x: os.path.join( actor_path, x ), os.listdir( actor_path ) )
    actress_files = map(lambda x: os.path.join( actress_path, x ), os.listdir( actress_path ) )

    if test == False:
        return actor_files[:-n_test], actress_files[:-n_test]
    else:
        return actor_files[-n_test:], actress_files[-n_test:]


def get_chairs(test=False, half=None, ver=360, angle_info=False):
    chair_ids = os.listdir( chair_path )
    if test:
        current_ids = chair_ids[-10:]
    else:
        if half is None: current_ids = chair_ids[:-10]
        elif half == 'first': current_ids = chair_ids[:-10][:len(chair_ids)/2]
        elif half == 'last': current_ids = chair_ids[:-10][len(chair_ids)/2:]

    chair_paths = []

    for chair in current_ids:
        current_path = os.path.join( chair_path, chair, 'renders' )
        if not os.path.exists( current_path ): continue
        filenames = filter(lambda x: x.endswith('.png'), os.listdir( current_path ))

        for filename in filenames:
            angle = int(filename.split('_')[3][1:])
            filepath = os.path.join(current_path, filename)

            if ver == 180:
                if angle > 180 and angle < 360: chair_paths.append(filepath)
            if ver == 360:
                chair_paths.append(filepath)

    return chair_paths




def get_faces_3d(test=False, half=None):
    files = os.listdir( face_3d_path )
    image_files = filter(lambda x: x.endswith('.png'), files)

    df = pd.DataFrame({'image_path': image_files})
    df['id'] = df['image_path'].map(lambda x: x.split('/')[-1][:20])
    unique_ids = df['id'].unique()

    if not test:
        if half is None:
            current_ids = unique_ids[:8]
        if half == 'first':
            current_ids = unique_ids[:4]
        if half == 'last':
            current_ids = unique_ids[4:8]
    else:
        current_ids = unique_ids[8:]

    groups = df.groupby('id')
    image_paths = []

    for current_id in current_ids:
        image_paths += groups.get_group(current_id)['image_path'].tolist()

    return image_paths
