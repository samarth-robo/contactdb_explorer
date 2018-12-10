from plyfile import PlyData
import matplotlib.pyplot as plt
import numpy as np
import urllib.request
import io
import plotly.offline as py
import plotly.graph_objs as go
import ipywidgets as widgets
from IPython.display import display

py.init_notebook_mode(connected=True)

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


class UI:
    def __init__(self, server_name='cayley.cc.gt.atl.ga.us'):
        self.server_name = server_name
        with open('use_object_names.txt', 'r') as f:
            use_object_names = [l.strip() for l in f]
        with open('handoff_object_names.txt', 'r') as f:
            handoff_object_names = [l.strip() for l in f]
        self.objects_widget = widgets.Dropdown(
            options=use_object_names,
            value=use_object_names[0],
            description='Object',
            disabled=False)
        self.instruction_widget = widgets.Dropdown(
            options=['use'],
            value='use',
            description='Intent',
            disabled=False)
        self.session_widget = widgets.IntSlider(
            value=4, min=1, max=42, step=1,
            description='Session',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True)
        self.show_button = widgets.Button(description='Show', disabled=False)
        self.show_button.on_click(self.show_object)
        self.fig = go.Figure(
            layout={'xaxis': {'visible': False, 'showspikes': False},
                    'yaxis': {'visible': False, 'showspikes': False},
                   }
        )
        display(self.objects_widget,
                self.session_widget,
                self.instruction_widget,
                self.show_button)

    def show_object(self, b):
        mesh_filename = 'http://{:s}/contactdb_dataset/full{:d}_{:s}_{:s}.ply'.\
            format(self.server_name, self.session_widget.value,
                   self.instruction_widget.value, self.objects_widget.value)
        
        mesh = read_ply(mesh_filename)
        self.fig.update({
            'data': [mesh],
        })
        py.iplot(self.fig)