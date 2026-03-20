# ///////// AUDIO /////////////

from src.ftm import rectangular_drum
from IPython.display import Audio, display
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colormaps
from scipy.io.wavfile import write

# Constants controlling the physical simulation of the drum and audio output.
constants = {
    "x1": 0.4,
    "x2": 0.4,
    "h": 0.03,
    "l0": np.pi,
    "m1": 10,
    "m2": 10,
    "sr": 22050,
    "dur":2**16
}
audio_path = "results/ways/audio/"

def audioFromNode(node, g=None):
    """
    Generate audio from a graph node or a list of raw parameters.

    If `node` is a string, the function looks up the node's 'features' attribute
    in graph `g`, parses it as a comma-separated list of floats, and uses those
    as synthesis parameters. If `node` is already a list of parameters, it is
    used directly.

    Parameters
    ----------
    node : str or list
        Either a node ID (string) from graph `g`, or a list of numeric parameters.
    g : networkx.Graph, optional
        The graph containing node feature data. Required if `node` is a string.

    Returns
    -------
    numpy.ndarray or None
        The synthesized audio signal as a NumPy array, or None if `g` is missing.
    """
    if type(node)==str:
        if g == None:
            print("Please provide the graph")
            return None
        # Parse the comma-separated feature string into a list of strings
        node_params = dict(g.nodes.data())[node]['features'].split(',')
        node_params = list(node_params)
    else :
        node_params = node
    
    # Prepend theta with [3] as a fixed leading parameter
    theta = []
    for x in node_params:
        theta.append(float(x))
    
    node_audio = rectangular_drum(theta, True, **constants)
    node_audio = np.array(node_audio)
    return node_audio

def playSequence(path,g=None, hop_time=1000,rec=False,recName=None):
    """
    Play a sequence of graph nodes as overlapping audio events.

    Each node in `path` is synthesized into audio and placed in a buffer
    at evenly spaced time offsets (hop_time). Successive sounds overlap if
    hop_time is shorter than the node's audio duration.

    Parameters
    ----------
    g : networkx.Graph
        The graph containing node feature data.
    path : list of str
        Ordered list of node IDs defining the sequence to play.
    hop_time : int, optional
        Time offset between successive notes in milliseconds. Default is 1000 ms.
    """
    node_song_duration = 65536  # Fixed duration per node in samples (2^16)
    hop_lenght = int(hop_time * constants["sr"] / 1000)  # Convert ms to samples
    N = len(path)
    
    # Allocate output buffer large enough for all notes with their offsets
    sequence = np.zeros(node_song_duration + hop_lenght * (N - 1))
    i = 0
    for node in path:
        # Synthesize and add audio for this node at the current offset
        node_audio = audioFromNode(node, g)
        sequence[hop_lenght*i:hop_lenght*(i) + node_song_duration] += node_audio
        i += 1

    #Save audio if needed
    if rec:
        if recName is None:
            recName = "Saved_audio"
        write(audio_path + recName + ".wav", constants["sr"], sequence)
    
    # Scale down to avoid clipping at output
    sequence = sequence * 0.9
    display(Audio(sequence, rate=constants["sr"]))

    
def list_neighbor(g, node):
    """
    List and play audio for a node and all its neighbors, sorted by edge weight.

    Retrieves the neighbors of `node` in graph `g`, sorts them by ascending
    edge weight (distance), and plays the audio for each one. Also plots
    the sorted weight distribution.

    Parameters
    ----------
    g : networkx.Graph or networkx.MultiGraph
        The graph to query. Handles both simple and multi-edge graphs.
    node : str
        The node ID whose neighborhood is to be explored.
    """
    n_list = list(g.neighbors(node))
    dist_list = []
    w = []
    
    for x_idx in range(len(n_list)):
        x = n_list[x_idx]
        # Handle both simple graphs (edges[u,v]) and multigraphs (edges[u,v,0])
        try:
            weight = g.edges[node, x]['weight']
        except:
            weight = g.edges[node, x, 0]['weight']
        
        w.append(weight)
        dist_list.append((weight, x))
    
    # Sort neighbors by ascending weight; prepend the node itself at distance 0
    n_list = sorted(dist_list)
    n_list = [(0.0, node)] + n_list
    print(n_list)
    sorted_neighbor = [x[1] for x in n_list]
    
    # Plot sorted edge weights to visualise the local distance distribution
    plt.figure()
    plt.plot(sorted(w), "+")
    plt.show()
    
    for nei_idx in range(len(sorted_neighbor)):
        print("Neighbor n°" + str(nei_idx) + " :" + str(n_list[nei_idx]))
        neighbor = sorted_neighbor[nei_idx]
        # Synthesise and play audio for each neighbor
        node_audio = audioFromNode(neighbor, g)
        display(Audio(node_audio, rate=22050))


# ///////// Graph analysis /////////////

from cdlib import algorithms

def edge_correction(g, edge_type):
    """
    Clean and optionally invert edge weights in-place.

    Removes edges with NaN or infinite weights. If `edge_type` is 'invDist',
    weights are replaced by their reciprocal (1 / (w + eps)), converting
    distances into similarities suitable for clustering algorithms that
    favour high-weight edges.

    Parameters
    ----------
    g : networkx.Graph
        The graph whose edges will be corrected (modified in-place).
    edge_type : str
        If 'invDist', weights are inverted. Any other value only removes
        invalid edges without transforming the weights.
    """
    if edge_type == "invDist":
        eps = 1e-10  # Small epsilon to avoid division by zero
        for data in list(g.edges(data=True)):
            if np.isnan(data[2]["weight"]) or data[2]["weight"] == np.inf:
                # NOTE: bug here — variable `edge` is undefined; should be `data`
                g.remove_edge(data[0], data[1])
            else:
                data[2]["weight"] = 1 / (data[2]["weight"] + eps)
    else:
        for edge in list(g.edges(data=True)):
            if np.isnan(edge[2]["weight"]) or edge[2]["weight"] == np.inf:
                g.remove_edge(edge[0], edge[1])


def get_idx_of_cluster_changes(g, path, clustering_technic="louvain"):
    """
    Return the indices in `path` where the community assignment changes.

    Clusters the graph using the specified technique, maps each node in `path`
    to its community ID, then identifies positions where consecutive nodes
    belong to different communities.

    Parameters
    ----------
    g : networkx.Graph
        The graph to cluster.
    path : list of str
        Ordered sequence of node IDs.
    clustering_technic : str, optional
        Clustering algorithm to use ('louvain', 'leiden', or 'walktrap').
        Default is 'louvain'.

    Returns
    -------
    list of int
        Indices i such that path[i-1] and path[i] are in different clusters.
    """
    # Perform clustering and map each node to its community index
    community_dico = clustering(g, clustering_technic, "invDist")
    cluster_path = [community_dico[node] for node in path]
    
    # Detect positions where the community label changes between consecutive nodes
    changes = []
    for i in range(1, len(cluster_path)):
        if cluster_path[i-1] != cluster_path[i]:
            changes.append(i)
    return changes


def get_idx_of_aggregation_changes(g, path, agregation):
    """
    Return the indices in `path` where the aggregation label changes.

    Reads an aggregation value for each node in `path` (from index 1 onward)
    using the numeric suffix of the node ID, then identifies positions where
    consecutive nodes have different aggregation labels.

    Parameters
    ----------
    g : networkx.Graph
        The graph (used for context; node IDs are expected as 'vN' strings).
    path : list of str
        Ordered sequence of node IDs with numeric suffixes (e.g. 'v0', 'v42').
    agregation : array-like
        Array indexed by node number, containing the aggregation label for each node.

    Returns
    -------
    list of int
        Indices i such that path[i] and path[i-1] have different aggregation labels.
    """
    # Extract the aggregation label for each node using its numeric ID suffix
    aggregation_list = []
    for k in range(1, len(path)):
        aggregation_list.append(int(agregation[int(path[k][1:])]))
    
    # Detect positions where the aggregation label changes
    aggregation_changes = []
    for i in range(1, len(aggregation_list)):
        if aggregation_list[i-1] != aggregation_list[i]:
            aggregation_changes.append(i)
    return aggregation_changes


def get_cumul_distances(g, path):
    """
    Compute cumulative edge-weight distances along a path in the graph.

    Parameters
    ----------
    g : networkx.Graph or networkx.MultiGraph
        The graph containing weighted edges.
    path : list of str
        Ordered sequence of node IDs forming the path.

    Returns
    -------
    list of float
        Cumulative distance at each node, starting at 0.0 for the first node.
        Length equals len(path).
    """
    cumul_dist = [0]
    for k in range(1, len(path)):
        # Handle both simple graphs and multigraphs
        try:
            dist = g.edges[path[k-1], path[k], 0]['weight']
        except:
            dist = g.edges[path[k-1], path[k]]['weight']
        cumul_dist.append(cumul_dist[-1] + dist)
    return cumul_dist


# Path to the folder containing KNN graph files
graph_path = "data/Knn-G"

def load_graph(graph_Name, edge_type='dist', verbose=True):
    """
    Load a graph from a GraphML file and apply edge correction.

    Reads a GraphML file from the standard graph directory, converts it to
    an undirected graph, and calls `edge_correction` to remove invalid edges
    (and optionally invert weights).

    Parameters
    ----------
    graph_Name : str
        Filename of the GraphML file (relative to the graphml_folder directory).
    edge_type : str, optional
        Edge weight handling: 'dist' keeps raw distances, 'invDist' inverts them.
        Default is 'dist'.
    verbose : bool, optional
        If True, prints a loading message. Default is True.

    Returns
    -------
    networkx.Graph
        The loaded and corrected undirected graph.
    """
    if verbose:
        print("loading " + graph_Name + "...")
    path = graph_path + "\\graphml_folder" + '\\' + graph_Name
    g = nx.read_graphml(path)
    g = g.to_undirected()
    edge_correction(g, edge_type)
    return g


# ///////// Graph plot /////////////

from matplotlib import cm

def connectedComponentsHisto(graph: nx.Graph, graph_Name, plot=True):
    """
    Compute (and optionally plot) the size distribution of connected components.

    Parameters
    ----------
    graph : networkx.Graph
        The graph to analyse.
    graph_Name : str
        Name of the graph, used as the plot title.
    plot : bool, optional
        If True, displays a bar chart of component sizes on a log scale.
        Default is True.

    Returns
    -------
    list of int
        Sorted list of connected component sizes.
    """
    components = nx.connected_components(graph)
    components_size = [len(c) for c in components]
    if plot:
        fig, ax = plt.subplots()
        ax.bar(np.linspace(0, len(components_size), len(components_size)), components_size)
        ax.set_yscale('log')
        ax.set_xlabel("Connected component")
        ax.set_ylabel("Number of nodes in the connected component")
        ax.set_title("Repartition of node in connected components for the graph : " + graph_Name)
        plt.show()
    return components_size


def plot_degree_histo(G, graph_name):
    """
    Plot the degree histogram of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The graph whose degree distribution is plotted.
    graph_name : str
        Name used in the plot title.

    Returns
    -------
    list of int
        Node degrees sorted in descending order.
    """
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    dmax = max(degree_sequence)
    fig = plt.figure("Degree of a random graph", figsize=(8, 8))
    axgrid = fig.add_gridspec(5, 4)
    ax2 = fig.add_subplot(axgrid[3:, 2:])
    ax2.bar(*np.unique(degree_sequence, return_counts=True))
    ax2.set_yscale('log')
    ax2.set_title("Degree histogram for graph : " + graph_name)
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("Number of Nodes")
    fig.tight_layout()
    plt.show()
    return degree_sequence

def plotSubGraph(g,node_list):
    #Extract subgraph
    G = g.subgraph(node_list)
    plt.figure()
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    edge_weights = [G[u][v].get('weight', 1) for u, v in G.edges()]  # Default to 1 if no weight is found
    
    # Create a colormap
    cmap = cm.grey
    
    # Normalize the edge weights for colormap
    norm = plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
    
    # Draw the graph
    pos = nx.spring_layout(G)  # Positions for all nodes
    edges = list(G.edges())
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500)

    # Draw edges with color mapped to edge weights
    nx.draw_networkx_edges(G, pos, edgelist=edges, width=2,
                           edge_color=edge_weights, edge_cmap=cmap, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights))
    
    plt.show()

# ///////// Graph clustering /////////////

# Supported clustering algorithms
clusteringTechnics = ["louvain", "leiden", "walktrap"]

# Output directory for clustering figures
pathFigure = "results/Clustering/"


def performClustering(g, clustering_technic):
    """
    Run a community detection algorithm on the graph.

    Parameters
    ----------
    g : networkx.Graph
        The graph to cluster.
    clustering_technic : str
        One of 'louvain', 'leiden', or 'walktrap'.

    Returns
    -------
    cdlib.NodeClustering or None
        The clustering result object, or None if the technique is unknown.
    """
    calculated_clustering = None
    if clustering_technic == "louvain":
        calculated_clustering = algorithms.louvain(g, weight='weight', resolution=1., randomize=False)
    elif clustering_technic == "leiden":
        calculated_clustering = algorithms.leiden(g,weights='weight')
    elif clustering_technic == "walktrap":
        calculated_clustering = algorithms.walktrap(g)
    else:
        print("unknow clustering technic")
        return None
    return calculated_clustering


def testGraphClustering(g, dico, verbose=False):
    """
    Benchmark all clustering techniques on a graph and store results.

    Runs each algorithm in `clusteringTechnics`, computes the modularity score,
    and appends results to `dico` under keys 'score<algo>' and 'nb_cluster<algo>'.

    Parameters
    ----------
    g : networkx.Graph
        The graph to evaluate.
    dico : dict
        Dictionary accumulating results across multiple calls (modified in-place).
    verbose : bool, optional
        If True, prints results for each algorithm. Default is False.
    """
    for clusteringTechnic in clusteringTechnics:
        calculated_clustering = performClustering(g, clusteringTechnic)
        score = nx.community.modularity(g, calculated_clustering.communities)
        
        # Append scores and community counts to the result dictionary
        if "score" + clusteringTechnic in dico and clusteringTechnic in dico :
            dico["score" + clusteringTechnic].append(score)
            dico["nb_cluster" + clusteringTechnic].append(len(calculated_clustering.communities))
            dico[clusteringTechnic].append(calculated_clustering.communities)
        else:
            dico["score" + clusteringTechnic] = [score]
            dico["nb_cluster" + clusteringTechnic] = [len(calculated_clustering.communities)]
            dico[clusteringTechnic] = [calculated_clustering.communities]
        
        if verbose:
            print("  Algo : " + clusteringTechnic)
            print("    Nulber of communities : " + str(len(calculated_clustering.communities)))
            print("    score = " + str(score))


def clustering(g_ref, clustering_technic, edge_type='dist'):
    """
    Cluster a graph and return a node-to-community mapping dictionary.

    Works on a copy of the input graph to avoid side effects.

    Parameters
    ----------
    g_ref : networkx.Graph
        The reference graph (not modified).
    clustering_technic : str
        One of 'louvain', 'leiden', or 'walktrap'.
    edge_type : str, optional
        Edge weight handling passed to `edge_correction`. Default is 'dist'.

    Returns
    -------
    dict
        Mapping from node ID (str) to community index (int).
    """
    # Work on a copy to avoid modifying the original graph
    g = g_ref.copy()
    edge_correction(g, edge_type)
    communities_list = performClustering(g, clustering_technic)
    
    # Build a flat node → community_index dictionary from the list of communities
    community_dico = {}
    for community_idx in range(len(communities_list.communities)):
        for node in communities_list.communities[community_idx]:
            community_dico[node] = community_idx
    return community_dico


def clustering_result_init(dico):
    """
    Initialise a result dictionary with empty lists for all clustering metrics.

    Should be called before passing `dico` to `testGraphClustering` for the
    first time so that keys are guaranteed to exist.

    Parameters
    ----------
    dico : dict
        Dictionary to initialise (modified in-place).
    """
    for clusteringTechnic in clusteringTechnics:
        dico["score" + clusteringTechnic] = []
        dico["nb_cluster" + clusteringTechnic] = []
        dico[clusteringTechnic] = []


def firstSigneComponent(nb_connected_component):
    """
    Return the index of the first entry equal to 1 in a list.

    Intended to find the value of k at which the graph first becomes
    fully connected (i.e. has exactly one connected component).

    Parameters
    ----------
    nb_connected_component : list of int
        List where each element is the number of connected components
        for a given k value.

    Returns
    -------
    int
        The index of the first occurrence of 1.
    """
    return nb_connected_component.index(1)

        
def plotScoring(clustering_result, k_bounds, idxSingleComponent=None):
    """
    Plot modularity scores for all clustering methods as a function of k.

    Parameters
    ----------
    clustering_result : dict
        Dictionary produced by repeated calls to `testGraphClustering`,
        containing 'score<algo>' lists for each algorithm.
    k_bounds : tuple of int
        (k_min, k_max) defining the range of k values on the x-axis.
    idxSingleComponent : int or None, optional
        If provided, draws a vertical dashed red line at this k value to
        indicate where the graph becomes fully connected. Default is None.
    """
    plt.figure()
    if idxSingleComponent:
        plt.axvline(idxSingleComponent, c="r", alpha=0.8, ls=':', label="Single connected component")
    
    k_list = np.linspace(k_bounds[0], k_bounds[1], k_bounds[1] - k_bounds[0] + 1)
    
    for clusteringTechnic in clusteringTechnics:
        plt.plot(k_list, clustering_result["score" + clusteringTechnic], label=clusteringTechnic)
    plt.legend()
    plt.title("Score of clustering for " + str(len(clusteringTechnics)) + " methods (Modularity index)")
    plt.xlabel("K")
    plt.ylabel("Score (Modularity index)")
    plt.savefig(pathFigure + "score_evolution.svg")
    plt.savefig(pathFigure + "score_evolution.png")
    plt.show()


def plotComponentsCurve(nb_connected_component, k_bounds, idxSingleComponent=None):
    """
    Plot the number of connected components as a function of k (log scale).

    Parameters
    ----------
    nb_connected_component : list of int
        Number of connected components for each k in the range.
    k_bounds : tuple of int
        (k_min, k_max) defining the range of k values on the x-axis.
    idxSingleComponent : int or None, optional
        If provided, draws a vertical dashed red line at this k value.
        Default is None.
    """
    plt.figure()
    if idxSingleComponent:
        plt.axvline(idxSingleComponent, c="r", alpha=0.8, ls=':', label="Single connected component")
    k_list = np.linspace(k_bounds[0], k_bounds[1], k_bounds[1] - k_bounds[0] + 1)
    plt.plot(k_list, nb_connected_component)
    plt.title("Number of connected components")
    plt.xlabel("K")
    # plt.yscale("log")
    plt.ylabel("Number of connected components")
    plt.savefig(pathFigure + "Number_of_connected_components_evolution.svg")
    plt.savefig(pathFigure + "Number_of_connected_components_evolution.png")
    plt.show()


def plotNbClusters(clustering_result, k_bounds, idxSingleComponent=None):
    """
    Plot the number of detected clusters for each algorithm as a function of k.

    Parameters
    ----------
    clustering_result : dict
        Dictionary produced by repeated calls to `testGraphClustering`,
        containing 'nb_cluster<algo>' lists for each algorithm.
    k_bounds : tuple of int
        (k_min, k_max) defining the range of k values on the x-axis.
    idxSingleComponent : int or None, optional
        If provided, draws a vertical dashed red line at this k value.
        Default is None.
    """
    plt.figure()
    if idxSingleComponent:
        plt.axvline(idxSingleComponent, c="r", alpha=0.8, ls=':', label="Single connected component")
    k_list = np.linspace(k_bounds[0], k_bounds[1], k_bounds[1] - k_bounds[0] + 1)
    for clusteringTechnic in clusteringTechnics:
        plt.plot(k_list, clustering_result["nb_cluster" + clusteringTechnic], label=clusteringTechnic)
    plt.legend()
    plt.title("Number of clusters given by three clustering methods")
    plt.xlabel("K")
    plt.ylabel("Number of clusters")
    plt.yscale("log")
    plt.savefig(pathFigure + "Number_of_clusters.svg")
    plt.savefig(pathFigure + "Number_of_clusters.png")
    plt.show()
