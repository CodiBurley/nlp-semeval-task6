import pickle
from sklearn.cluster import DBSCAN
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
import numpy as np
import pandas

df = pandas.read_csv('Subtask B/donaldTrumpTweets', sep='\t', encoding='latin1')
df = df.loc[df['Tweet'] != 'Not Available']
df = df['Tweet']

X_tsne = None
with open('tsne.pickle', 'rb') as handle:
    X_tsne = pickle.load(handle)

db = DBSCAN(eps=3.5, min_samples = 2000, n_jobs = -1).fit(np.array(X_tsne))
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

output_file("DBSCAN.html")

# p = figure(title = "DBSCAN",tools="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,save")

colormap = {-1: 'black', 0: 'red', 1: 'green', 2: 'blue'}
colors = [colormap[x] for x in labels]

source = ColumnDataSource(data=dict(
    x=X_tsne[:,0],
    y=X_tsne[:,1],
    colors=colors,
    desc=list(df),
))

hover = HoverTool(tooltips=[
    ("index", "$index"),
    ("desc", "@desc"),
])

p = figure(tools=[hover,'pan','reset','wheel_zoom'], title="Mouse over the dots",plot_width=1000,plot_height=800)

p.circle('x', 'y', color='colors', fill_alpha=0.2, size=7, source=source)

show(p)


# for i, point in enumerate(X_tsne):
#     col = (0,1,1,1)
#     if labels[i] == -1:
#         col = colors[0]
#     elif labels[i] == 0:
#         col = colors[1]
#     elif labels[i] == 1:
#         col = colors[2]
#     elif labels[i] == 2:
#         col = colors[3]

#     plt.plot(point[0], point[1], 'o', markerfacecolor=col,
#              markeredgecolor='k', markersize=6)# for k, col in zip(unique_labels, colors):
# plt.show()
