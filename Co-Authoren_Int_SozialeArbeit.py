# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:30:11 2017

@author: Makrus Eckl

Title: 'The Mil. Effect '
With: Christian Gahnem and Heiko Löwenstein

"""

#%%
import pandas as pd
import numpy as np
from numpy import *
import re
import itertools as it
import matplotlib.pyplot as plt
import networkx as nx
import powerlaw


###


# I. Create the strcture of the data

def data_str_peron(dirname):

    df = pd.read_csv(dirname, sep=',', header=0)
    last_index = -1
    result = []
    d = None
    progress = 0
    for row in df.iterrows():
        index = row[1]["articleID"]
        if index != last_index:
            if d is not None:     #????
                result.append(d)
            d = { "index": index }
        attribute = row[1]["attributes"]
        value = row[1]["record"]
        if attribute in ["author","authorAff"]:
            if not attribute in d:
                d[attribute]= []
            d[attribute].append(value) #keyWords werden nicht berücksichigt (gibt ja auch mehrere)
        else:
            d[attribute] = value
        last_index = index
        progress += 1
        if progress % 10000 == 0:
            print(progress)

    #print(result[0])

    article = []
    data_structure = []
    for i in result:
        if ('author' in i.keys())  and ('article' in i.keys()) and ('pubYear' in i.keys()):
            article.append(i['article'])
            #print(article)
            data_structure.append(i)

    df_data = pd.DataFrame.from_dict(data_structure)
    df_data.to_csv("df_soc_work_int.csv", sep='\t', encoding='utf-8')
    #print(df_data.head())
    transfer = [data_structure, result]

    return(transfer)

transfer = data_str_peron('C:\Markus_Eckl\Data\Social_Work_peron_data\Social Work Research Database V2.csv')

data = transfer[0]
rohdata = transfer[1]





print('N numbers of used article:',len(data))



# II. Distribution of Co-authorship articles
def distribution_article_auhtor(data_str):

    co_author = []
    year_co_author = []
    year_co_author_count = []
    one_author = []
    year_one_author = []
    year_one_author_count = []
    for i in data_str:
        if (('author') in i.keys()) and (('pubYear') in i.keys()):
            if len(i['author']) > 1:
                co_author.append(i)
                year_co_author.append(i['pubYear'])
            if len(i['author']) == 1:
                one_author.append(i)
                year_one_author.append(i['pubYear'])
    for i in year_co_author:
        year_co_author_count.append(year_co_author.count(i))
    for i in year_one_author:
        year_one_author_count.append(year_one_author.count(i))

    print('N numbers of used article with Co-Author', len(co_author))
    print('N numbers of article with one author:', len(one_author))

    distribution_co =  sorted(list(set(zip(year_co_author, year_co_author_count))))
    distribution_one = sorted(list(set(zip(year_one_author, year_one_author_count))))
    print(distribution_one)
    print(distribution_co)
    #Plot
    x_val = [x[0] for x in distribution_co]
    y_val = [x[1] for x in distribution_co]
    x_val2 = [x[0] for x in distribution_one]
    y_val2 = [x[1] for x in distribution_one]
    with plt.style.context('fivethirtyeight'):
        plt.plot(x_val,y_val)
        plt.plot(x_val,y_val,'or')
        plt.plot(x_val2,y_val2)
        plt.plot(x_val2,y_val2,'or')

    plt.legend(['Co-Authorship','Article with one author'])
    plt.xlabel('Year')
    plt.ylabel('N Article')
    plt.show()

    return(co_author)

data_co_author = distribution_article_auhtor(data)


# III. Time structure of the data
def time_class(co_author_structure):

    art_90er_1 = []
    art_90er_2 = []
    art_00er_1 = []
    art_00er_2 = []
    art_10er_1 = []
    art_all = []

    for v in co_author_structure:
        if (('pubYear') in v.keys()):
            if (v['pubYear'] >= '1990') and (v['pubYear'] <= '1994'):
                art_90er_1.append(v)
            elif (v['pubYear'] >= '1995') and (v['pubYear'] <= '1999'):
                art_90er_2.append(v)
            elif (v['pubYear'] >= '2000') and (v['pubYear'] <= '2004'):
                art_00er_1.append(v)
            elif (v['pubYear'] >= '2005') and (v['pubYear'] <= '2009'):
                art_00er_2.append(v)
            elif (v['pubYear'] >= '2010') and (v['pubYear'] <= '2014'):
                art_10er_1.append(v)
            if (v['pubYear'] >= '1990') and (v['pubYear'] <= '2014'):
                art_all.append(v)

    print('N numbers of Co-Aut. 1990-2014:', len(art_all), '\n',
          'N numbers of Co-Aut. 1990-1994 Jahre:', len(art_90er_1), '\n',
          'N numbers of Co-Aut. 1995-1999 Jahre:', len(art_90er_2), '\n',
          'N numbers of Co-Aut. 2000-2004 Jahre:', len(art_00er_1), '\n',
          'N numbers of Co-Aut. 2005-2009 Jahre:', len(art_00er_2), '\n',
          'N numbers of Co-Aut. 2010-2014 Jahre:', len(art_10er_1))

    transfer2 = [art_90er_1, art_90er_2, art_00er_1, art_00er_2, art_10er_1, art_all]

    return(transfer2)

transfer2 = time_class(data_co_author)
#


# IV. Build the Networks
def name_cleaning (author_list):
    new = []
    for a in author_list:
        for u in a:
            clean_author = [u.upper().strip() for u in a]
        new.append(clean_author)

    for i in new:
        try:
            i.remove('ET')
        except ValueError:
            pass


    return(new)

def netzwerk_building(data_str_for_network):
    #MultiGraph
    MG = nx.MultiGraph()
    G = nx.Graph()
    authors = []
    for i in data_str_for_network:
        authors.append(i['author'])

    clean_author = name_cleaning(authors)

    for aut in clean_author:
        for i in it.combinations(aut, 2):
            MG.add_edges_from([i])
            G.add_edges_from([i])


    #Subgraph of the MultiGraph
    sub_graphs = list(nx.connected_component_subgraphs(MG))
    cur_graph = sub_graphs[0]
    for graph in sub_graphs:
        if len(graph.nodes()) > len(cur_graph.nodes()):
            cur_graph = graph
    biggest_sup = cur_graph
    MG_sub = biggest_sup

    components_MG = [len(graph.nodes()) for graph in nx.connected_component_subgraphs(MG)]


    #Subgraph of the Graph
    sub_graphs_G = list(nx.connected_component_subgraphs(G))
    cur_graph = sub_graphs_G[0]
    for graph in sub_graphs_G:
        if len(graph.nodes()) > len(cur_graph.nodes()):
            cur_graph = graph
    biggest_sup = cur_graph
    G_sub = biggest_sup

    print('N number of nodes MG', len(MG.nodes()))
    print('N number of edges MG', len(MG.edges()))

    print('N number of nodes MG_sub', len(MG_sub.nodes()))
    print('N number of edges MG_sub', len(MG_sub.edges()))
    print ('N numbers of components:', len(components_MG))

    print('N number of nodes G', len(G.nodes()))
    print('N number of edges G', len(G.edges()))
    print('N number of nodes G_sub', len(G_sub.nodes()))
    print('N number of edges G_sub', len(G_sub.edges()))


    transfer_graph = [MG, MG_sub, G, G_sub]

    return(transfer_graph)


# V. Network properties
print('---------------- Graphs_90er_1 --------------------')
Graphs_90er_1  = netzwerk_building(transfer2[0])
print('---------------- Graphs_90er_2 --------------------')
Graphs_90er_2  = netzwerk_building(transfer2[1])
print('---------------- Graphs_00er_1 --------------------')
Graphs_00er_1 = netzwerk_building(transfer2[2])
print('---------------- Graphs_00er_2 --------------------')
Graphs_00er_2 = netzwerk_building(transfer2[3])
print('---------------- Graphs_10er_1 --------------------')
Graphs_10er_1  = netzwerk_building(transfer2[4])
print('---------------- Graphs_all --------------------')
Graphs_all = netzwerk_building(transfer2[5])


# VI. Network analysis: Degree distribution
def network_degree(multi_graph, graph_sub):

    #Degree
    degree_graph = nx.degree(multi_graph)
    print('mean degree', mean(list(degree_graph.values())))
    print('percentile 95% ', np.percentile(list(degree_graph.values()), 95))
    print('mean of the top 10 ', mean(sorted(list(degree_graph.values()),reverse = True)[:10]))

    degree_sub_g = nx.degree(graph_sub)
    mean_degree_sub_g = mean(list(degree_sub_g.values()))
    print('mean degree sub_graph', mean(list(degree_sub_g.values())))

    print('----------------sw-----------------------')
    degree_sub_graph = nx.degree(graph_sub)
    k = mean(list(degree_sub_graph.values()))
    k2 = int(round(k+0.5))
    print('k2', k2)
    len_sub_nodes_gr = len(graph_sub.nodes())
    print('len_sub_nodes_gr', len_sub_nodes_gr)
    SW = nx.watts_strogatz_graph(len_sub_nodes_gr, k2, 0.1)
    print( 'average clustering coefficient random Graph:', nx.average_clustering(SW))
    print( 'average clustering coefficient empirical Graph:', nx.average_clustering(graph_sub))
    print('average shortest path length random Graph:', nx.average_shortest_path_length(SW))
    print('average shortest path empirical Graph:', nx.average_shortest_path_length(graph_sub))

    return (degree_graph, mean_degree_sub_g)




print('---------------- Graphs_90er_1 --------------------')
degree_sw_Graphs_90er_1 = network_degree(Graphs_90er_1[0], Graphs_90er_1[3])
print('---------------- Graphs_90er_2 --------------------')
degree_sw_Graphs_90er_2 = network_degree(Graphs_90er_2[0], Graphs_90er_2[3])
print('---------------- Graphs_00er_1 --------------------')
degree_sw_Graphs_00er_1 = network_degree(Graphs_00er_1[0], Graphs_00er_1[3])
print('---------------- Graphs_00er_2 --------------------')
degree_sw_Graphs_00er_2 = network_degree(Graphs_00er_2[0], Graphs_00er_2[3])
print('---------------- Graphs_10er_1 --------------------')
degree_sw_Graphs_10er_1 = network_degree(Graphs_10er_1[0], Graphs_10er_1[3])
print('---------------- Graphs_all --------------------')
degree_sw_Graphs_all = network_degree(Graphs_all[0], Graphs_all[3])




# VII. Power Law distribution of the degree
degree_all = nx.degree(Graphs_all[0])
fit = powerlaw.Fit(list(degree_all.values()))
print('power law alpha:', fit.power_law.alpha, '\n',
      'power law sigma:', fit.power_law.sigma, '\n',
      'distribution compare:', fit.distribution_compare('power_law','exponential'), '\n',
       'min:' ,  fit.xmin)
#Fig1:
powerlaw.plot_pdf(list(degree_all.values()), color='b')
plt._show()
#Fig2:
fig2 = fit.plot_pdf(color= 'b', linewidth=2)
fit.power_law.plot_pdf(color='b', linestyle='--', ax=fig2)
fit.plot_ccdf(color='r', linewidth=2, ax=fig2)
fit.power_law.plot_ccdf(color='r', linestyle='--', ax=fig2)
plt._show()

# VIII. Visualisation in Gephi
MG_sub2_1990er = nx.MultiGraph()
MG_sub2_1990er = nx.compose(Graphs_90er_1[1], Graphs_90er_2[1])
MG_sub2_2000er = nx.MultiGraph()
MG_sub2_2000er = nx.compose(Graphs_00er_1[1], Graphs_00er_2[1])

nx.write_graphml(MG_sub2_1990er, "MG_sub2_1990er.graphml")
nx.write_graphml(MG_sub2_2000er, "MG_sub2_2000er.graphml")
nx.write_graphml(Graphs_10er_1[1], "MG_sub2_2010er_1.graphml")


#VIII: Visualistion
data1 = [('1990', 452), ('1991', 501), ('1992', 432), ('1993', 473), ('1994', 521), ('1995', 495), ('1996', 495), ('1997', 545), ('1998', 509), ('1999', 604), ('2000', 580), ('2001', 634), ('2002', 630), ('2003', 665), ('2004', 689), ('2005', 692), ('2006', 649), ('2007', 744), ('2008', 699), ('2009', 750), ('2010', 804), ('2011', 790), ('2012', 765), ('2013', 712), ('2014', 710), ('2015', 582) ]
data2 = [('1990', 289), ('1991', 301), ('1992', 372), ('1993', 395), ('1994', 383), ('1995', 411), ('1996', 391), ('1997', 435), ('1998', 432), ('1999', 479), ('2000', 519), ('2001', 584), ('2002', 629), ('2003', 683), ('2004', 755), ('2005', 805), ('2006', 824), ('2007', 930), ('2008', 883), ('2009', 1067), ('2010', 1149), ('2011', 1191), ('2012', 1227), ('2013', 1316), ('2014', 1252), ('2015', 1157)]
df1 = pd.DataFrame(data1, columns=["year", "articles_single"])
df2 = pd.DataFrame(data2, columns=["year2", "articles_co_aut"])
df3 = pd.concat([df1, df2], axis=1)
del df3['year2']
df3["articles_single_rf"] = df3.articles_single / ((df3.articles_single)+(df3.articles_co_aut))
df3["articles_co_aut_rf"] = df3.articles_co_aut / ((df3.articles_single)+(df3.articles_co_aut))
plt.style.context('seaborn-whitegrid')
df3.plot(x="year", y=["articles_single_rf", "articles_co_aut_rf"])
plt.show()

