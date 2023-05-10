"""
Practica 2  - climex
"""
#------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p2/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p2/salidas/'
save = False
dpi = 100
#------------------------------------------------------------------------------#
# Funciones
def SelectAreas(df, low, top, n_col=0):
    return df.loc[(df[df.columns[n_col]]>=low)&(df[df.columns[n_col]]<=top)]

def SelEstacion(ds, num_est):
    return ds.where(ds.estacion==num_est, drop=True)

def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)

def CheckFaltantes(data, variable, save=True):
    if variable != 'ppc':
        aux = data.loc[np.isnan(data[variable])]
        if save:
            np.savetxt(out_dir + variable + '_falantes.txt', aux, fmt='%s')
        return aux
    else:
        aux = data.loc[data['ppc'].notnull()]
        if save:
            np.savetxt(out_dir + variable + '_falantes.txt', aux, fmt='%s')
        return aux
#------------------------------------------------------------------------------#
data = pd.read_csv(data_dir + 'estaciones_1959_2020.csv', sep=',', header=None)
data_2021 = pd.read_csv(data_dir + 'estaciones_2021.csv', sep=',', header=None)
#------------------------------------------------------------------------------#
# Region: PAMPA 30-40S 63-55W
#Selección de estaciones
estaciones = pd.read_csv('/home/luciano.andrian/doc/climaext/p1/data/'
                         'Estaciones_lista.csv')
aux = SelectAreas(estaciones, -63, -55) #en longitud
est_sel = SelectAreas(aux, -40, -30, n_col=1) #en longitud
#------------------------------------------------------------------------------#
for num in est_sel['OMM']:
    est = pd.concat(
        [data.loc[data[0]==num], data_2021.loc[data_2021[0]==num]], axis=0)

    if len(est)!=0: # no estan todas las estaciones de la practica anterior
        est.columns = \
            ['estacion', 'dia', 'mes', 'anio', 'tx', 'tm', 'pp', 'ppc', 'h',
             'dv', 'vx']

        # datos faltantes:
        # Temp  = Nan,
        # PP = nan & ppc ~= nan |  <<<<<<<<<<CONFIRMAR!>>>>>>>>>>>>
        tx_faltantes = CheckFaltantes(est, 'tx', save=False)
        tm_faltantes = CheckFaltantes(est, 'tm', save=False)
        pp_faltantes = CheckFaltantes(est, 'ppc', save=False)
        # control tx > tm
        tm_tx = est.loc[est['tx'] < est['tm']]

        d = {'estacion': [num],
             'tx_faltantes': [len(tx_faltantes)],
             'tm_faltantes': [len(tm_faltantes)],
             'pp_faltantes': [len(pp_faltantes)],
             'tm_tx': [len(tm_tx)]}

        if num == 87374: #la primera
            df = pd.DataFrame(d)
        else:
            df = df.append(pd.DataFrame(d), ignore_index=True)

# estaciones disponibles Parana y Junin.
# Parana: muchos datos faltantes de tx al comienzo de registro + 3 de pp
# Junin: pocos datos faltantes, pero falta un mes entero por tx<tm
jn = pd.concat(
    [data.loc[data[0]==87548], data_2021.loc[data_2021[0]==87548]], axis=0)
jn.columns = ['estacion', 'dia', 'mes', 'anio', 'tx', 'tm', 'pp', 'ppc', 'h',
             'dv', 'vx']

tx_faltantes = CheckFaltantes(jn, 'tx', False)
tm_faltantes = CheckFaltantes(jn, 'tm', False)
pp_faltantes = CheckFaltantes(jn, 'ppc', False)
tm_tx = jn.loc[jn['tx'] < jn['tm']]
np.savetxt(out_dir + 'tm_tx_falantes.txt', tm_tx, fmt='%s')

# eliminando el unico valor faltante de pp
jn = jn.drop(pp_faltantes.index)

# todo julio de 1963 es cualquier cosa en todas las variables
aux = jn.loc[(jn['mes']==7) & (jn['anio']==1963)]
jn = jn.drop(aux.index)

# buscando otra vez
tm_tx = jn.loc[jn['tx'] < jn['tm']]
# 1963: confuso. eliminar esa fecha
jn = jn.drop(tm_tx.index[0])
# 1984: conservar maxima, la minima podria considerarse 13.3 en lugar de 213.3
jn.loc[(jn['mes']==2) & (jn['anio']==1984) & (jn['dia']==7)].index[0]
index = jn.loc[jn.loc[(jn['mes']==2) & (jn['anio']==1984) &
                      (jn['dia']==7)].index[0], 'tm']
jn.loc[index, 'tm']=13.3

#------------------------------------------------------------------------------#
# Calcular los percentiles diarios (1, 10, 90, 99) de tx y tm y los
# percentiles mensuales (95, 99) de la pp (no nula) para el período 1981-2010.

jn_81_10 = jn.loc[(jn['anio']>=1981)&(jn['anio']<=2010)]

# temp
for q in [.01, .10, .90, .99]:
    for mes in range(1,13):
        for dia in range(1,32):
            if (mes==2) and (dia==29):
                pass
            else:
                per = jn_81_10.loc[(jn['dia'] == dia) &
                                   (jn['mes'] == mes)].quantile(q)

                if np.isnan(per.tx):
                    pass
                else:
                    d = {'dia':[dia], 'mes':[mes],
                         'tx_' + str(q).split('.')[1]:[per.tx],
                         'tm_' + str(q).split('.')[1]:[per.tm]}

                    if dia == 1:
                        dias = pd.DataFrame(d)
                    else:
                        dias = pd.concat([dias, pd.DataFrame(d)], axis=0)

        if mes == 1:
            meses = dias
        else:
            meses = pd.concat([meses, dias], axis=0)

    if q == .01:
        perc_t = meses
    else:
        perc_t = pd.concat([perc_t, meses[[meses.columns[2],meses.columns[3]]]]
                           , axis=1)

# pp
# acumulado mensual de cada año
for a in range(1981,2011):
    for mes in range(1,13):
        ppa = jn_81_10.loc[(jn['anio'] == a) &
                           (jn['mes'] == mes)].sum().pp

        m = {'mes':[mes], 'anio':[a], 'ppa':[ppa]}
        if mes == 1:
            meses = pd.DataFrame(m)
        else:
            meses = pd.concat([meses, pd.DataFrame(m)], axis=0)

    if a == 1981:
        ppaf = meses
    else:
        ppaf = pd.concat([ppaf, meses], axis=0)

#percentiles
for q in [.95,.99]:
    for mes in range(1, 13):
        per = ppaf.loc[ppaf['mes'] == mes].quantile(q)

        m = {'mes':[mes], 'pp_'+ str(q).split('.')[1]:[per.ppa]}
        if mes == 1:
            meses = pd.DataFrame(m)
        else:
            meses = pd.concat([meses, pd.DataFrame(m)], axis=0)

    if q == .95:
        per_pp = meses
    else:
        per_pp = pd.concat([per_pp, meses], axis=1)



