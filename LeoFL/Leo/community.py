# 위성 간 인접행렬로 그래프를 만들어 커뮤니티(군집)를 탐지·시각화하는 유틸(networkx).
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def update_community_visualiztion(adj_matrix, fig, ax):
    G = nx.from_numpy_array(adj_matrix)
    communities = list(nx.community.girvan_newman(G))
    # print(communities)

    # Modularity -> measures the strength of division of a network into modules
    modularity_df = pd.DataFrame(
        [
            [k + 1, nx.community.modularity(G, communities[k])]
            for k in range(len(communities))
        ],
        columns=["k", "modularity"],
    )
    ax[0].clear()
    ax[1].clear()
    visualize_communities(G, communities[0], ax[0])
    # visualize_communities(G, communities[1], ax[1])

    # Plot change in modularity as the important edges are removed
    modularity_df.plot.bar(
        x="k",
        ax=ax[1],
        color="#F2D140",
        title="Modularity Trend for Girvan-Newman Community Detection",
    )
    fig.canvas.draw()
    # fig.canvas.flush_events()


#    return communities


# function to create node colour list
def create_community_node_colors(graph, communities):
    # number_of_colors = len(communities)
    colors = [
        "#D4FCB1",
        "#CDC5FC",
        "#FFC2C4",
        "#F2D140",
        "#BCC6C8",
        "#196f3d",
        "#76448a",
        "#b03a2e",
        "#b9770e",
    ]  # [:number_of_colors]
    node_colors = []
    for node in graph:
        current_community_index = 0
        for community in communities:
            if node in community:
                if node == 0:
                    node_colors.append("green")
                else:
                    node_colors.append(colors[current_community_index])
                break
            current_community_index += 1
    return node_colors


# function to plot graph with node colouring based on communities
def visualize_communities(graph, communities, ax):
    node_colors = create_community_node_colors(graph, communities)
    modularity = round(nx.community.modularity(graph, communities), 6)
    title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
    pos = nx.spring_layout(graph, k=0.3, iterations=50, seed=2)
    # pos = nx.planar_layout(graph)
    # plt.subplot(3, 1, i)
    # plt.title(title)
    nx.draw(
        graph,
        pos=pos,
        node_size=300,
        node_color=node_colors,
        with_labels=True,
        font_size=8,
        font_color="black",
        ax=ax,
    )


if __name__ == "__main__":
    # Load karate graph and find communities using Girvan-Newman
    G = nx.karate_club_graph()
    communities = list(nx.community.girvan_newman(G))

    print(len(communities))
    # Modularity -> measures the strength of division of a network into modules
    modularity_df = pd.DataFrame(
        [
            [k + 1, nx.community.modularity(G, communities[k])]
            for k in range(len(communities))
        ],
        columns=["k", "modularity"],
    )

    fig, ax = plt.subplots(3, figsize=(15, 20))

    # Plot graph with colouring based on communities
    visualize_communities(G, communities[0], 1)
    visualize_communities(G, communities[1], 2)

    # Plot change in modularity as the important edges are removed
    modularity_df.plot.bar(
        x="k",
        ax=ax[2],
        color="#F2D140",
        title="Modularity Trend for Girvan-Newman Community Detection",
    )
    plt.show()
