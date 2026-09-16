
import os
import shutil
import sys

if len(sys.argv) != 3:
    print(
        "Invalid argument count! Please pass source directory and destination directory after the file name"
    )
    sys.exit()


current_path = os.getcwd()
grandparent_path = "/".join(current_path.split("/")[:-1])

print("Looking for modules in : ", grandparent_path)


f = open("files_to_sync.txt", "r+")
files_to_move = f.read().splitlines()
f.close()


source = sys.argv[1]
dest = sys.argv[2]


try:
    for file in files_to_move:
        print(f"Moving file : ", file)
        shutil.copy(
            os.path.join(grandparent_path, source, file),
            os.path.join(grandparent_path, dest, file),
        )
    print(f"Finished moving {len(files_to_move)} files")
except Exception as e:
    print(
        "Something went wrong! please check if the source and destination folders are present in same folder"
    )
