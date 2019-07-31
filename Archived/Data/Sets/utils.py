import os
import shutil


# use if with file name source folder
def copy_files_to_dir(files, sdir, tdir):
    for f in files:
        f = os.path.join(sdir, f)
        shutil.copy(f, tdir)


# create dir if it does not exists
def create_output_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# delete dir
def cleandir(dirpath):
    if os.path.exists(dirpath):
        print(f'cleaning dir: {dirpath}')
        shutil.rmtree(dirpath)
