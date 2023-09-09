
import pandas as pd, numpy as np
from IPython.display import display,HTML

def build_dataframe_p_value(dict_v, nm_model, col='K'):
    df = pd.DataFrame(pd.Series(dict_v)).reset_index().rename(columns={'index': col})
    df[col] = pd.to_numeric(df[col])
    df = df.sort_values(col)
    
    cols = []
    for c in df.columns:
        if c != col:
            cols.append(f'{nm_model}')
        else:
            cols.append(c)
        
    df.columns = cols
    return df

def build_dataframe_driver_license(d):
    res = pd.DataFrame()
    for k in d.keys():
        res = res.append(pd.DataFrame(pd.Series(d[k].PI, name=f'Policy {k}')).T)
        
    return res

def multi_column_df_display(list_dfs, cols=2):
    html_table = "<table style='width:100%; border:0px'>{content}</table>"
    html_row = "<tr style='border:0px'>{content}</tr>"
    html_cell = "<td style='width:{width}%;vertical-align:top;border:0px'>{{content}}</td>"
    html_cell = html_cell.format(width=100/cols)

    cells = [ html_cell.format(content=df.to_html()) for df in list_dfs ]
    cells += (cols - (len(list_dfs)%cols)) * [html_cell.format(content="")] # pad
    rows = [ html_row.format(content="".join(cells[i:i+cols])) for i in range(0,len(cells),cols)]
    display(HTML(html_table.format(content="".join(rows))))
    
def build_dataframe_all_point(dict_metrica, model1, model2, met1):
    df_allpoint = pd.DataFrame()
    
    for i in dict_metrica.keys():
        d = dict_metrica[i]
        for v in d.items():
            min_key = min(d, key=d.get)
            if v[1] == d[min_key]:
                append = pd.DataFrame([i, v[0], v[1]]).T
                df_allpoint = df_allpoint.append(append)
                
    df_allpoint.columns = [model1, model2, met1]
    df_allpoint = df_allpoint[df_allpoint[met1] != np.inf]
    return df_allpoint

def build_dataframes(d1, d2, d3, met1, met2, model1, model2):
    res1 = build_dataframe_p_value(d1, met1, model1)
    res2 = build_dataframe_p_value(d2, met2, model1)
    res3 = build_dataframe_p_value(d3, f'{model2}_MAX', model1)

    res4 = res1.merge(res2, on=model1)
    res4 = res4.merge(res3, on=model1)

    try:
        res4[f'DIF_{met2}'] = 1 + (abs(res4[f'{met1}']) - \
                  abs(res4[f'{met2}']))/res4[f'{met1}']
    except:
        print('Não foi possível capturar a diferença.')

    res4 = res4.dropna()
    
    return res1, res2, res3, res4