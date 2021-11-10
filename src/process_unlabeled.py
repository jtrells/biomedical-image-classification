# +
import glob
import os
from PIL import Image
from tqdm import tqdm
import argparse
import csv
from functools import partial
from multiprocessing import Pool
from tqdm.contrib.concurrent import process_map  # or thread_map

BASE_PATH = '/mnt'


# -

def get_txt_route(path):
    if path.split('/')[-1].startswith('0'):
        ind = path.split('/')[-2]
        return os.path.join('/'.join(path.split('/')[:-2]),f'{ind}.txt')

    else:
        ind = path.split('/')[-1].split('.')[0]
        return os.path.join('/'.join(path.split('/')[:-1]),f'{ind}.txt')


# +
def check_type_files(file_list,file_type = ['.jpg']):
    all_files = []
    for file in file_list:
        if file.endswith(tuple(file_type)):
            all_files.append(file)
    return all_files

def get_size_image(path):
    image = Image.open(path)
    return image.size

def extract_txt(path):
    route = get_txt_route(path)
    try:
        f = open(route, mode="r", encoding="utf-8")
        return [i for i in f][0]
    except:
        return ' '


# -

def write_folder_to_csv(output_path,source,path_folder,error_path):
    with open(output_path, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for root, dirs, files in os.walk(path_folder, topdown=False):
            if len(files)>0:
                image_files = check_type_files(files,file_type = ['.jpg'])
                if image_files:
                    for image_element in image_files:
                        total_path    = os.path.join(root,image_element)
                        try:
                            width, height = get_size_image(total_path)
                            img_path    = os.path.join(root,image_element).replace('/mnt/','')
                            caption       = extract_txt(total_path)
                            row = ['img',source,img_path,caption,width,height]
                            writer.writerow(row)
                        except:
                            print(total_path)
                            with open('error.txt', 'a+') as f:
                                f.write("%s\n" % total_path)


# +
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocessing Dataset")
    parser.add_argument("--path"   , type = str, default = 'cord19-uic')
    parser.add_argument("--source" , type = str, default = 'cord19')
    parser.add_argument("--output" , type = str, default = 'test.csv')
    parser.add_argument("--error" , type = str, default = 'error.txt')
    args = parser.parse_args()
    return args

def Write2CSV_mp(folder,output_path,path,source,error_path):
    path_folder = os.path.join(BASE_PATH,path,folder)
    write_folder_to_csv(output_path,source,path_folder,error_path)



# -

if __name__== '__main__':
    args = parse_args()
    PATH        = args.path #'cord19-uic'
    SOURCE      = args.source #'cord-19'
    OUTPUT_PATH = args.output #'test.csv'
    ERROR_PATH  = args.error #'error.txt'
    folder = [os.path.join(BASE_PATH,PATH,folder) for folder in os.listdir(os.path.join(BASE_PATH,PATH))]
    with open(OUTPUT_PATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['img','source', 'img_path','caption','width','height'])
    
    process_map(partial(Write2CSV_mp, 
                  output_path = OUTPUT_PATH,
                  path        = PATH,
                  source      = SOURCE,
                  error_path  = ERROR_PATH),
                  folder, max_workers = 8)

# +
#import pandas as pd
#pd.read_csv('test.csv')

# +
'''output_path,source = 'TEST_MP',SOURCE
def Write2CSV(path):
    #list_error = []
    for folder in tqdm(os.listdir(os.path.join(BASE_PATH,path))[5245:52800]):
        path_folder = os.path.join(BASE_PATH,path,folder)
        write_folder_to_csv(output_path,source,path_folder)
    #print(list_error) 
    #with open('error.txt', 'w') as f:
    #    for item in list_error:
    #        f.write("%s\n" % item)'''

"""
if __name__== '__main__':
    
    PATH        = 'cord19-uic'
    SOURCE      = 'cord-19'
    OUTPUT_PATH = 'test.csv'
    ERROR_PATH  = 'error.txt'
    folder = [os.path.join(BASE_PATH,PATH,folder) for folder in os.listdir(os.path.join(BASE_PATH,PATH))]

    p = Pool(5)
    p.imap(partial(Write2CSV_mp, 
                  output_path = OUTPUT_PATH,
                  path        = PATH,
                  source      = SOURCE,
                  error_path  = ERROR_PATH),
           folder)
"""

