import shutil
import os


def delete_empty_folders(root):

    deleted = set()
    
    for current_dir, subdirs, files in os.walk(root, topdown=False):

        still_has_subdirs = False
        for subdir in subdirs:
            if os.path.join(current_dir, subdir) not in deleted:
                still_has_subdirs = True
                break
    
        if not any(files) and not still_has_subdirs:
            os.rmdir(current_dir)
            deleted.add(current_dir)

    return deleted


dst = '../simulations_simple/simulations'
shutil.copytree('../simulations', dst, ignore=shutil.ignore_patterns('*.jpg', '*.png'))
delete_empty_folders(dst)