"""
Practica 4  - climex
"""
#------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
#------------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p2/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p4/salidas/'
save = False
dpi = 100
#------------------------------------------------------------------------------#
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<preproc de p2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
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
print('preproc p2...')
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
jn['tm']=jn['tm'].replace([213.3], 13.3)
# buscando otra vez
tm_tx = jn.loc[jn['tx'] < jn['tm']]
#nada
#ok
print('done preproc p2')
#------------------------------------------------------------------------------#
# fin
# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<preproc de p2>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#------------------------------------------------------------------------------#

"""
mes calido 4/2021, frio 5/2021, humedo 3/2020
usar mmonth en pronos y calcular acumulado de pp y promedio de t en el trimestre
de mmonth
"""
observado = []
for mm, y, pp in zip([4,5,3], [2021,2021,2020], [False, False, True]):
    aux_y = jn.loc[jn['anio'] == y]
    if mm < 3:
        break
        # de todos modos no hay ninguno

    aux_mm = aux_y.loc[(aux_y['mes'] >= mm - 1) & (aux_y['mes'] <= mm + 1)]

    if pp:
        obs = np.sum(aux_mm['pp'])
    else:
        obs = aux_mm[aux_mm.columns[4:5]].mean().values[0]

    observado.append(obs)

print('T mean MAM 2021: ', observado[0], 'ºC')
print('T mean AMJ 2021: ', observado[1], 'ºC')
print('PP acum. FMA 2021: ', observado[2], 'mm')
print('done')
#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#