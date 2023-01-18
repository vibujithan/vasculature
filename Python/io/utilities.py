import glob
import json
import math
import os


def read_config(config_path, session_to_process=None, lasers_to_process=None, pzf_dirs_to_process=None):
    f = open(config_path, "r")
    config = json.loads(f.read())
    f.close()

    root_dir = os.path.dirname(config_path)
    no_of_sessions = len(config['Sessions'])

    stp = range(*(session_format(session_to_process)).indices(no_of_sessions))

    sessions = {}
    idx = 0
    for s in stp:
        sessions[idx] = {}
        print(f'Imaging session: {s}')
        session = config['Sessions'][s]
        s_id = session['Session']
        s_dir = os.path.join(root_dir, s_id)

        lasers = session['Laser Sequence']
        print(f'\tFound laser sequences: {lasers}')

        ll = laser_format(lasers_to_process)
        ltp = ll if ll else lasers

        sessions[idx]['Files'] = {}
        for l in ltp:
            files = sorted(glob.glob(os.path.join(s_dir, f"*{l}*")))
            pzf_dir = list(filter(os.path.isdir, files))
            print(f'\t{l}: {len(pzf_dir)} Blocks with {len(os.listdir(pzf_dir[0]))} planes')
            sessions[s]['Files'][l] = pzf_dir[pzf_dir_format(pzf_dirs_to_process)]

        sessions[idx]['Image Start'] = session['YScan']['Image Start']
        sessions[idx]['Image End'] = session['YScan']['Image End']
        sessions[idx]['Pixel Size'] = session['YScan']['Pixel Size']
        sessions[idx]['Scale Factor'] = session['YScan']['Scale Factor']

        img_res_y = 1e6 * session['YScan']['Pixel Size'] * session['YScan']['Scale Factor']
        img_res_z = 1e6 * session['ZScan']['Z Increment']

        sessions[idx]['Image Resolution'] = (img_res_z, 0.23325, img_res_y)

        overlap = round(2048 - abs(1e6 * session['XScan']['X Increment']) / 0.23325)
        sessions[idx]['Overlap'] = overlap
        idx += 1

    return sessions


def get_files(dirctory, z_range=''):
    block = {'h5_filepath': os.path.join(os.path.dirname(dirctory), 'h5', f'{os.path.basename(dirctory)}.h5')}

    if z_range == '':
        zr = slice(None)
    elif '-' in z_range:
        zr = slice(*map(int, z_range.split('-')))
    else:
        print('Invalid z range')
        return 0

    block['pzf_files'] = sorted(glob.glob(os.path.join(dirctory, "*.pzf")))[zr]
    return block


def input_format(string):
    if '-' in string:
        return slice(*map(int, string.split('-')))
    else:
        return int(string)


def opt_chunksize(depth, xy_chunks=(256, 256), lower=200, upper=400):
    if depth < lower:
        chunk_d = depth
    else:
        divisor = 1
        chunk_d = depth
        while not (lower <= chunk_d < upper):
            divisor += 1
            chunk_d = math.ceil(depth / divisor)

    return (chunk_d,) + xy_chunks


def session_format(string):
    if string == "":
        return slice(None)
    elif '-' in string:
        return slice(*map(int, string.split('-')))
    else:
        num = int(string)
        return slice(num, num + 1, 1)


def laser_format(string):
    if string == "":
        return 0
    elif ',' in string:
        return string.split(',')
    else:
        return [string]


def pzf_dir_format(string):
    if string == "":
        return slice(None)
    elif '-' in string:
        return slice(*map(int, string.split('-')))
    else:
        num = int(string)
        return slice(num, num + 1, 1)


def get_save_path(project_name="CV"):
    f = open("../configs/save_path.json", "r")
    paths = json.loads(f.read())
    f.close()

    return paths[project_name]
