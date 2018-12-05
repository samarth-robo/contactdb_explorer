from plyfile import PlyData
import matplotlib.pyplot as plt
import numpy as np

def texture_proc(tex, sigmoid_k=10.0, max_frac=0.75):
    idx = tex > 0
    ci = tex[idx]
    thresh = max_frac * np.max(ci)
    ci = np.exp(sigmoid_k * (ci-thresh)) / (1 + np.exp(sigmoid_k * (ci-thresh)))
    tex[idx] = ci
    colors = plt.cm.inferno(tex)[:, :3]
    return colors

def read_ply(filename, sigmoid_k=10.0, max_frac=0.75):
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
    x = np.asarray(plydata['vertex'].data['x'])
    y = np.asarray(plydata['vertex'].data['y'])
    z = np.asarray(plydata['vertex'].data['z'])
    tex = np.asarray(plydata['vertex'].data['red'])
    
    vertices = np.vstack((x, y, z)).T
    faces = np.vstack(plydata['face'].data['vertex_indices'])
    colors = texture_proc(tex/255.0, sigmoid_k=sigmoid_k,
                          max_frac=max_frac)
    
    return vertices, faces, colors