from plyfile import PlyData
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import io
import plotly.graph_objs as go

def texture_proc(tex, sigmoid_k=10.0, max_frac=0.75):
    idx = tex > 0
    ci = tex[idx]
    thresh = max_frac * np.max(ci)
    ci = np.exp(sigmoid_k * (ci-thresh)) / (1 + np.exp(sigmoid_k * (ci-thresh)))
    tex[idx] = ci
    colors = plt.cm.inferno(tex)[:, :3]
    return colors

def read_ply(filename, sigmoid_k=10.0, max_frac=0.75):
  with urllib.request.urlopen(filename) as r:
  # with open(filename, 'rb') as r:
    f = io.BufferedReader(io.BytesIO(r.read()))
  plydata = PlyData.read(f)
  x = np.asarray(plydata['vertex'].data['x'])
  y = np.asarray(plydata['vertex'].data['y'])
  z = np.asarray(plydata['vertex'].data['z'])
  tex = np.asarray(plydata['vertex'].data['red'])
  
  vertices = np.vstack((x, y, z)).T
  faces = np.vstack(plydata['face'].data['vertex_indices'])
  colors = texture_proc(tex/255.0, sigmoid_k=sigmoid_k,
                        max_frac=max_frac)
  
  mesh = go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
               i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
  mesh['flatshading'] = False
  mesh['vertexcolor'] = colors
  
  return mesh
