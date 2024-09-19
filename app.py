from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from collections import Counter
import re
import networkx as nx
import itertools
from gensim import corpora
from gensim.models import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis



# 讀取PTT股票版的資料
df = pd.read_json('./data/pttstock_processed.json')
df['Date'] = pd.to_datetime(df['Date'])

# 讀取台股加權指數的資料
df_twse = pd.read_csv('./data/merged_taiex_data.csv')

# 資料的確認
print("原始資料：")
print(df_twse.head())
print(df_twse.dtypes)
print(df_twse.isnull().sum())

# 日期轉換
df_twse['Date'] = pd.to_datetime(df_twse['Date'], errors='coerce')

# 數值資料的轉換  
numeric_columns = ['Open', 'Max', 'Min', 'Close']
for col in numeric_columns:
    df_twse[col] = pd.to_numeric(df_twse[col].str.replace(',', ''), errors='coerce')

# NaN的處理
df_twse = df_twse.dropna(subset=['Date'] + numeric_columns)

# 資料的再確認
print("\n處理後的資料：")
print(df_twse.head())
print(df_twse.dtypes)
print(df_twse.isnull().sum())

# 日期排序
df_twse = df_twse.sort_values('Date')

# 蠟燭圖的製作
fig = go.Figure(data=[go.Candlestick(x=df_twse['Date'],
                open=df_twse['Open'],
                high=df_twse['Max'],
                low=df_twse['Min'],
                close=df_twse['Close'])])

# 排版設定
fig.update_layout(
    title='台股加權指數',
    xaxis_title='日期',
    yaxis_title='指數',
    xaxis_rangeslider_visible=True
)

# 設定視覺主題
external_stylesheets = [dbc.themes.FLATLY]
app = Dash(__name__, external_stylesheets=external_stylesheets)

# 自訂CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>PTT股票版分析</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background-color: #f0f0f0;
                font-family: 'Noto Sans TC', sans-serif;
            }
            .container {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
                padding: 20px;
                margin-top: 20px;
            }
            .graph-container {
                background-color: white;
                border-radius: 10px;
                box-shadow: 0 0 5px rgba(0,0,0,0.1);
                padding: 15px;
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1('PTT股票版熱詞與台灣加權指數分析', className="text-primary text-center mb-4"),
            html.Div([
                html.Label('選擇發文日期：', className="mr-2"),
                dcc.Dropdown(
                    df['Date'].dt.strftime('%Y-%m-%d').unique(),
                    '請選擇發文日期',
                    id='xaxis-column',
                    className="mb-3"
                )
            ], style={'width': '50%', 'margin': '0 auto'})
        ])
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(figure=fig)
            ], className="graph-container")
        ], width=12),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id='word-frequency-graph')
            ], className="graph-container")
        ], width=6),
        dbc.Col([
            html.Div([
                dcc.Graph(id='co-occurrence-network')
            ], className="graph-container")
        ], width=6),
    ]),
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H3("主題模型結果", className="text-center"),
                html.Div(id='topic-modeling-result')
            ], className="graph-container")
        ], width=12),
    ])
], fluid=True, className="mt-4")

# 停止詞讀取（必要時調整）
stopwords = set(STOPWORDS)

# 更新圖表樣式的函數
def update_graph_style(fig):
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Noto Sans TC, sans-serif"),
        title=dict(font=dict(size=24)),
        legend=dict(
            bgcolor='rgba(255,255,255,0.5)',
            bordercolor='rgba(0,0,0,0.1)',
            borderwidth=1
        )
    )
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1ヶ月", step="month", stepmode="backward"),
                dict(count=6, label="6ヶ月", step="month", stepmode="backward"),
                dict(count=1, label="1年", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')
    return fig

# 主題模型化函數
def perform_topic_modeling(texts, num_topics=5):
    # 單詞的標記化
    tokenized_texts = [text.split() for text in texts]
    
    # 辭典的製作
    dictionary = corpora.Dictionary(tokenized_texts)
    
    # 語料庫的製作
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    
    # LDA模型訓練
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
    
    return lda_model, corpus, dictionary

# 主題的可視化
def visualize_topics(lda_model, corpus, dictionary):
    vis_data = gensimvis.prepare(lda_model, corpus, dictionary)
    return pyLDAvis.prepared_data_to_html(vis_data)

@callback(
    Output('word-frequency-graph', 'figure'),
    Input('xaxis-column', 'value'))
def update_word_frequency(selected_date):
    if selected_date is None or selected_date == '請選擇發文日期':
        return go.Figure()

    filtered_df = df[df['Date'].dt.strftime('%Y-%m-%d') == selected_date]
    
    # 將所有名詞結合為串列
    all_nouns = ' '.join(filtered_df['nouns']).split()
    
    # 套用停止詞
    nouns = [noun for noun in all_nouns if noun not in stopwords]
    
    # 計算名詞的頻度
    noun_counts = Counter(nouns)
    
    # 取得頻度上位20的名詞
    top_nouns = noun_counts.most_common(20)
    
    # 轉換為數據框
    noun_freq_df = pd.DataFrame(top_nouns, columns=['noun', 'frequency'])

    fig = px.bar(
        noun_freq_df,
        x='noun',
        y='frequency',
        title=f'{selected_date}的名詞詞頻前20名',
        labels={'noun': '名詞', 'frequency': '詞頻'}
    )

    return update_graph_style(fig)

@callback(
    Output('co-occurrence-network', 'figure'),
    Input('xaxis-column', 'value'))
def update_co_occurrence_network(selected_date):
    if selected_date is None or selected_date == '請選擇發文日期':
        return go.Figure()

    filtered_df = df[df['Date'].dt.strftime('%Y-%m-%d') == selected_date]
    
    # 將所有名詞結合為串列
    all_nouns = ' '.join(filtered_df['nouns']).split()
    
    # 套用停止詞
    valid_nouns = [noun for noun in all_nouns if noun not in stopwords]
    
    # 共起語的頻度計算
    noun_pairs = list(itertools.combinations(valid_nouns, 2))
    co_occurrence = Counter(noun_pairs)
    
    # 繪製網路
    G = nx.Graph()
    for (noun1, noun2), count in co_occurrence.most_common(50):  # 上位50的共起關係使用
        G.add_edge(noun1, noun2, weight=count)
    
    # 計算節點的位置
    pos = nx.spring_layout(G)
    
    # 繪製邊線
    edge_x = []
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    # 繪製節點
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            )
        ),
        text=[],
        textposition="top center"
    )

    # 設定節點的顏色和文本
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'{adjacencies[0]} - {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    # 製作佈局
    layout = go.Layout(
        title=f'{selected_date}的名詞共現網路',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))

    # 製作圖表
    fig = go.Figure(data=[edge_trace, node_trace], layout=layout)

    return update_graph_style(fig)

# 更新台灣加權指數圖表的樣式
fig = update_graph_style(fig)

# 添加回調函數
@callback(
    Output('topic-modeling-result', 'children'),
    Input('xaxis-column', 'value'))
def update_topic_modeling(selected_date):
    if selected_date is None or selected_date == '請選擇發文日期':
        return "請選擇發文日期"

    filtered_df = df[df['Date'].dt.strftime('%Y-%m-%d') == selected_date]
    
    # 執行主題模型化
    lda_model, corpus, dictionary = perform_topic_modeling(filtered_df['nouns'], num_topics=5)
    
    # 主題的可視化
    vis_html = visualize_topics(lda_model, corpus, dictionary)
    
    return html.Iframe(srcDoc=vis_html, width='100%', height='800px')

if __name__ == '__main__':
    app.run(debug=True)
    