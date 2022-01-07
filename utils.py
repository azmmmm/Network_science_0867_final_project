import pandas as pd
import gzip
import json
import matplotlib.pyplot as plt
import powerlaw
import matplotlib.patches as mpatches
import copy
import networkx as nx
import http.client, urllib.request, urllib.parse, urllib.error, base64
import numpy as np


def getDF(data,dtype=None):
  i = 0
  df = {}
  for d in data:
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index',dtype=dtype)


def Timeseries_datapoints_API(headers,reporting_economies_str,partner_economies_str,meta='False',max=100,timePeriod='all',offset=0):
    """
    reporting_economies_str,
    partner_economies_str,
    meta='False',max=100,
    timePeriod='all',offset=0
    """
    #i:indicators info
    """
        {
        "code": "HS_M_0020",
        "name": "Bilateral imports by MTN product category",
        "categoryCode": "TPM_HS",
        "categoryLabel": "Bilateral imports",
        "subcategoryCode": null,
        "subcategoryLabel": null,
        "unitCode": "USD",
        "unitLabel": "US$",
        "startYear": 1996,
        "endYear": 2020,
        "frequencyCode": "A",
        "frequencyLabel": "Annual",
        "numberReporters": 139,
        "numberPartners": 191,
        "productSectorClassificationCode": "MT2",
        "productSectorClassificationLabel": "Tariffs - Multilateral Trade Negotiations Product Categories",
        "hasMetadata": "No",
        "numberDecimals": 0,
        "numberDatapoints": 3026578,
        "updateFrequency": "Every two months",
        "description": null,
        "sortOrder": 606
        }
        """
    params = urllib.parse.urlencode({
        # Request parameters
        #i:indicators
        

        'i': 'HS_M_0020',

        #汇报的经济体
        'r':reporting_economies_str,

        #Partner economies
        'p':partner_economies_str,
        
        #offset
        'off':offset,
        #Time period
        'ps':timePeriod,
        'mode':'codes',
        #一次最多返回的数据量
        'max':max,

        'meta':meta

    })

    try:
        conn = http.client.HTTPSConnection('api.wto.org')
        conn.request("GET", "/timeseries/v1/data?%s" % params, "{body}", headers)
        response = conn.getresponse()
        data = response.read()
        json_data=data.decode('utf8')
        data=json.loads(json_data)
        conn.close()
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))

    return data
    ####################################



def get_data(timePeriod,headers,reporting_economies_str,partner_economies_str,offset=0,step=10000,end=None,data_size=None):
    #先获取数据的信息
    if data_size==None:
        meta_data=Timeseries_datapoints_API(headers,reporting_economies_str,partner_economies_str,meta='True',max=1,timePeriod=timePeriod)
        data_size=meta_data['Summary'][0]['DataSet'][0]['TotalCount']

    data=[]
    while True:
        if end!=None and offset>=end:
            break
        print(f'collecting {offset}/{data_size}...     \t ',end='\r')
        data_page=Timeseries_datapoints_API(headers,reporting_economies_str,partner_economies_str,meta='False',max=step,timePeriod=timePeriod,offset=offset)
        if data_page.get('status'):
            #204 no content
            data_page['status']==204
            break
        data+=data_page['Dataset']
        offset+=step
    return data

"""
#old version
country_codes=pd.read_csv('./data/countries_code.csv',dtype=str)
country_codes.drop(['displayOrder'],inplace=True,axis=1)
country_codes.set_index(['code'],inplace=True)
_codes_dict=country_codes.to_dict()
def country_code_to_name(code):
    global _codes_dict
    return _codes_dict['name'][code],_codes_dict['iso3A'][code]
"""

country_info=pd.read_csv('./data/countries_info.csv',dtype=str)
_codes_dict={}
for i in range(0,len(country_info)):
    code=country_info['code'][i]
    iso3A=country_info['iso3A'][i]
    name=country_info['name'][i]
    latitude=country_info['latitude'][i]
    longitude=country_info['longitude'][i]
    _codes_dict[code]={
        'iso3A':iso3A,
        'name':name,
        'latitude':latitude,
        'longitude':longitude
        }
del country_info

def country_code_to_info(code):
    """
    return iso3A,name,latitude,longitude
    """
    global _codes_dict
    d=_codes_dict[code]
    return d['iso3A'],d['name'],float(d['latitude']),float(d['longitude'])

def annual_graph(selectedProducts,data_names):
    """
    每年数据单独生成一张图
    """
    for data_name in data_names:
        data=pd.read_csv(f'./data/{data_name}',dtype=str)
        data.drop(['IndicatorCategoryCode','IndicatorCode','ProductOrSectorClassificationCode','PeriodCode','FrequencyCode','UnitCode','ValueFlagCode'],inplace=True,axis=1)
        graph=nx.DiGraph()
        add_data_to_graph(graph,data,selectedProducts)
        #nx.write_edgelist(graph,f'./graph/{str(selectedProducts)}_{data_name[5:9]}.edgelist')
        nx.write_gexf(graph,f'./graph/{str(selectedProducts)}_{data_name[5:9]}.gexf')

def total_graph(selectedProducts,data_names):
    """所有数据生成一张图"""
    graph=nx.DiGraph()
    for data_name in data_names:
        data=pd.read_csv(f'./data/{data_name}',dtype=str)
        data.drop(['IndicatorCategoryCode','IndicatorCode','ProductOrSectorClassificationCode','PeriodCode','FrequencyCode','UnitCode','ValueFlagCode'],inplace=True,axis=1)
        
        add_data_to_graph(graph,data,selectedProducts)
        
    nx.write_gexf(graph,f'./graph/{str(selectedProducts)}_total.gexf')

def add_data_to_graph(graph,data,selectedProducts):
    for i in range(0,len(data)):
            importer=data['ReportingEconomyCode'][i]
            exporter=data['PartnerEconomyCode'][i]
            
            im_iso3A,im_name,latitude,longitude=country_code_to_info(importer)
            graph.add_node(importer,name=im_name,iso3A=im_iso3A,latitude=latitude,longitude=longitude)

            ex_iso3A,ex_name,latitude,longitude=country_code_to_info(exporter)
            graph.add_node(exporter,name=ex_name,iso3A=ex_iso3A,latitude=latitude,longitude=longitude)

            graph.add_edge(exporter,importer)
            
            ##总贸易额
            if graph.edges[exporter,importer].get('totalValue'):
                graph.edges[exporter,importer]['totalValue']+=float(data['Value'][i])

                graph.nodes[importer]['totalImportValue']+=float(data['Value'][i])
                graph.nodes[exporter]['totalExportValue']+=float(data['Value'][i])
            else:
                graph.edges[exporter,importer]['totalValue']=float(data['Value'][i])

                graph.nodes[importer]['totalImportValue']=float(data['Value'][i])
                graph.nodes[exporter]['totalExportValue']=float(data['Value'][i])
            ##选定产品的贸易额
            if data['ProductOrSectorCode'][i]in selectedProducts:
                if graph.edges[exporter,importer].get('selectedValue'):
                    graph.edges[exporter,importer]['selectedValue']+=float(data['Value'][i])

                    graph.nodes[importer]['selectedImportValue']+=float(data['Value'][i])
                    graph.nodes[exporter]['selectedExportValue']+=float(data['Value'][i])
                else:
                    graph.edges[exporter,importer]['selectedValue']=float(data['Value'][i])
                    graph.nodes[importer]['selectedImportValue']=float(data['Value'][i])
                    graph.nodes[exporter]['selectedExportValue']=float(data['Value'][i])

def fit_powerloaw_draw(degrees,name,xmin=5):
    plt.xlabel("k")
    plt.ylabel("P(k)")
    
    degree_sequence = sorted(degrees, reverse=True)
    fit = powerlaw.Fit(degree_sequence,xmin=xmin,discrete = True) 
    fig=fit.plot_pdf(color='b', linewidth=2)
    fit.power_law.plot_pdf(color='g', ax=fig)
    fit.lognormal.plot_pdf(color='r', ax=fig)

    red_patch = mpatches.Patch(color='red',linestyle='--', label=f"fit.lognormal.mu={fit.lognormal.mu:{4}.{4}}")
    green_patch = mpatches.Patch(color='g',linestyle='--', label=f'fit.powerlaw.alpha={fit.power_law.alpha:{4}.{4}}')
    blue_patch = mpatches.Patch(color='b',linestyle='--', label='origin_data')
    plt.legend(handles=[red_patch, green_patch,blue_patch])

    plt.title(name+f'\ntruncted={xmin}')

def degree_distribution_bar(graph,name,degree):
    plt.figure()
    #get distribution
    degree_distribution={}
    for i in graph:
        if degree==None:
            i_degree=graph.degree(i)
        elif degree=='in':
            i_degree=graph.in_degree(i)
        elif degree=='out':
            i_degree=graph.out_degree(i)
        if(degree_distribution.get(i_degree)):
            degree_distribution[i_degree]+=1
        else:
            degree_distribution[i_degree]=1

    #show plot
    data_size=max(degree_distribution)
    bar_x=[i+1 for i in range(data_size)]
    bar_y=np.zeros(data_size)
    for i in degree_distribution:
        bar_y[i-1]=degree_distribution[i]
    plt.bar(bar_x,bar_y,width=5)
    plt.title(f"{name} degree distreibution")
    plt.xlabel("degree")
    plt.ylabel("number of nodes")
    plt.show()

def degree_distribution(graph,name,degree=None):
    
    degree_distribution_bar(graph,name,degree)

    degrees=[d for n,d in graph.degree]    
    #fit_powerloaw_draw(degrees,f'degree distribution')

def add_partition_tag(G,partition,partition_name):
    for i,p in zip(range(1,len(partition)+1),partition):
        for node in list(p):
            G.nodes()[node][partition_name]=float(i)

def avg_node_weight(graph,weight,default=0):
    """
    当节点不含对应权重时，默认值为default
    """
    data=[]
    for node in graph.nodes:
        if graph.nodes[node].get(weight):
            data.append(float(graph.nodes[node][weight]))
        else:
            data.append(default)
    return sum(data)/len(data)

def top10_analisis(graph):
    frames=[]
    #入度
    degrees=list(graph.in_degree())
    degrees.sort(key=lambda x:(x[1]),reverse=True)
    top10=pd.DataFrame(degrees,columns=['name','in_degree'])[0:10]
    for i in range(0,len(top10)):
        code=top10.loc[i,'name']
        _,name,_,_=country_code_to_info(code)
        top10.loc[i,'name']=name
    frames.append(top10)

    #出度
    degrees=list(graph.out_degree())
    degrees.sort(key=lambda x:(x[1]),reverse=True)
    top10=pd.DataFrame(degrees,columns=['name','out_degree'])[0:10]
    for i in range(0,len(top10)):
        code=top10.loc[i,'name']
        _,name,_,_=country_code_to_info(code)
        top10.loc[i,'name']=name
    frames.append(top10)

    #聚类系数
    clustering=[]
    for key,value in nx.clustering(graph).items():
        clustering.append((key,value))
    clustering.sort(key=lambda x:(x[1]),reverse=True)

    top10=pd.DataFrame(clustering,columns=['name','cluster_coe'])[0:10]
    for i in range(0,len(top10)):
        code=top10.loc[i,'name']
        _,name,_,_=country_code_to_info(code)
        top10.loc[i,'name']=name
    frames.append(top10)

    #平均进出口额
    static_values=['totalExportValue','totalImportValue','selectedExportValue','selectedImportValue']

    for s_value in static_values:
        values=[]
        for node in graph:
            if graph.nodes[node].get(s_value):
                node_value=graph.nodes[node][s_value]
            else:
                node_value=0
            values.append((node,node_value))
        values.sort(key=lambda x:(x[1]),reverse=True)

        top10=pd.DataFrame(values,columns=['name',s_value])[0:10]
        for i in range(0,len(top10)):
            code=top10.loc[i,'name']
            _,name,_,_=country_code_to_info(code)
            top10.loc[i,'name']=name
        frames.append(top10)

    return pd.concat(frames,axis=1)