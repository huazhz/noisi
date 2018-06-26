import os
import io
import time
import json
from noisi import _ROOT

def setup_proj(project_name):

    os.makedirs(os.path.join(project_name))
    
    with io.open(os.path.join(_ROOT,'config','config.json'),'r+') as fh:
        conf = json.loads(fh.read())
        
    conf['date_created'] = time.strftime("%Y.%m.%d")
    conf['project_name'] = project_name
    conf['project_path'] = os.path.abspath(project_name)
    conf['wavefield_path'] = os.path.abspath(project_name)+'/path_to_wavefield'
    
    with io.open(os.path.join(project_name,'config.json'),'w') as fh:
        cf = json.dumps(conf,sort_keys=True, indent=4, separators=(",", ": "))
        fh.write(cf)