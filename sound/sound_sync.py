import os
import subprocess
import time

def sound(file_path):
    DEVNULL = open(os.devnull, "w")
    subprocess.call(["amixer", "set", "100%"], stdout=DEVNULL)
    subprocess.Popen(["aplay", file_path])
    DEVNULL.close()
    return True

    
    