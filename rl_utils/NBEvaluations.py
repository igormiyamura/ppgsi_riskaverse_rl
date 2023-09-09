
import numpy as np, pandas as pd, time

# ---------------------------------------------------------
# -- Policy Evaluation of different pre-determined policies

def run_driving_license(dl, T, C, actions, model, model_name, **kwargs):
    if model_name == 'Neutral-VI' or model_name == 'Neutral-PI':
        epsilon, discount_factor = kwargs['epsilon'], kwargs['discount_factor']
        rdl = model(dl, T, C, actions, discount_factor=discount_factor, epsilon=epsilon)
    elif model_name == 'EUF_RSVI':
        vl_lambda, epsilon = kwargs['vl_lambda'], kwargs['epsilon']
        rdl = model(dl, T, C, vl_lambda, actions, epsilon=epsilon)
    elif model_name == 'EUF_RSPI':
        vl_lambda, epsilon, explogsum = kwargs['vl_lambda'], kwargs['epsilon'], kwargs['explogsum']
        rdl = model(dl, T, C, vl_lambda, actions, epsilon=epsilon, explogsum=explogsum)
    elif model_name == 'PL_RSVI':
        k, alpha, gamma, epsilon = kwargs['k'], kwargs['alpha'], kwargs['gamma'], kwargs['epsilon']
        rdl = model(dl, T, C, k=k, alpha=alpha, gamma=gamma,
                         num_actions=actions, epsilon=epsilon)
    
    if 'PI' in kwargs:
        acc_costs = rdl.calculate_value_for_policy(kwargs['PI'], 0)
        return acc_costs
    else:
        num_iterations, V, PI = rdl.run_converge()
    
    if kwargs['_log']: print(f'Número de Iterações: {num_iterations}......')
    
    return rdl

def run_policies_evaluation(dl, T, C, actions, f, policies, range_values, normalize, **kwargs):
    res = {}
    df_res = pd.DataFrame()

    for policy in policies.keys():
        PI = policies[policy]
        res[policy] = {}

        for i in range(0, len(range_values)):
            print(f'{policy} | {range_values[i]}...', end='\r')
            if kwargs['model_name'] == 'EUF_RSVI': res[policy][range_values[i]] = f(dl, T, C, actions, PI=PI, vl_lambda=range_values[i], **kwargs)
            elif kwargs['model_name'] == 'PL_RSVI': 
                range_alpha = kwargs['range_alpha']
                res[policy][range_values[i]] = f(dl, T, C, actions, PI=PI, k=range_values[i], alpha=range_alpha[i], **kwargs)
            
        df = pd.DataFrame(pd.Series(res[policy], name=policy))
        df_res[policy] = df

    if normalize:
        for l in df_res.index:
            df_res.loc[l, :] = np.log(df_res.loc[l, :])/l
        
    return df_res

# -------------------------------------------------------------
# -- Policy Evaluation differences between algorithm's policies

def get_PEXP(dl, T, C, epsilon, model, vl_lambda, PI, explogsum=False):
    if vl_lambda == 0: return np.inf
    
    if hasattr(model, 'explogsum'):
        m = model(dl, T, C, vl_lambda, epsilon=epsilon, explogsum=explogsum)
    else:
        m = model(dl, T, C, vl_lambda, epsilon=epsilon)
        
    return m.calculate_value_for_policy(PI, vl_lambda)

def get_PLIN(dl, T, C, epsilon, model, k, PI):
    vl_k, vl_gamma, vl_alpha = k[0], k[1], k[2]
    m = model(dl, T, C, vl_k, vl_alpha, vl_gamma, epsilon=epsilon)
    return m.calculate_value_for_policy(PI, vl_k)

def comparing_policy_value(dl, T, C, epsilon, f, p1, p2, model, model_type1, model_type2):
    res_model1 = {}
    for i in p1.keys():
        if model_type1 == 'EXP':
            value1 = i
        if model_type1 == 'LIN':
            value1 = i[0]
            vl_gamma = i[1]
            vl_alpha = i[2]

        res_model1[str(value1)] = \
        f(dl, T, C, epsilon, model, i, p1[i].PI)

    res_model1_model2 = {}
    res_model1_model2_max = {}
    res_met1_max = {}

    for i in p1.keys():
        if model_type1 == 'EXP':
            value1 = i
        if model_type1 == 'LIN':
            value1 = i[0]
            vl_gamma = i[1]
            vl_alpha = i[2]
        
        max_k, last_dif = np.inf, np.inf

        for j in p2.keys():
            if model_type2 == 'EXP':
                value2 = j
            if model_type2 == 'LIN':
                value2 = j[0]
                vl_gamma = j[1]
                vl_alpha = j[2]

            if not str(value1) in res_model1_model2.keys():
                res_model1_model2[str(value1)] = {}
            
            res_model1_model2[str(value1)][str(value2)] = \
                    f(dl, T, C, epsilon, model, i, p2[j].PI)

            VALUE_EXP_LIN = res_model1_model2[str(value1)][str(value2)]
            VALUE_EXP = res_model1[str(value1)]
            dif = abs(VALUE_EXP_LIN - VALUE_EXP)

            if dif < last_dif:
                max_k = value2
                last_dif = dif

        try:
            res_model1_model2_max[str(value1)] = res_model1_model2[str(value1)][str(max_k)]
            res_met1_max[str(value1)] = max_k
        except:
            res_model1_model2_max[str(value1)] = np.inf
            res_met1_max[str(value1)] = None

    return res_model1, res_model1_model2, res_model1_model2_max, res_met1_max