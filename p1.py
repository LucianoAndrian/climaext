#-----------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
#-----------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p1/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p1/'
save = False
dpi = 300
#-----------------------------------------------------------------------------#
def selecteras(df, low, top, n_col=0):
    return df.loc[(df[df.columns[n_col]]>=low)&(df[df.columns[n_col]]<=top)]
#-----------------------------------------------------------------------------#
# Region: PAMPA 30-40S 63-55W
#-----------------------------------------------------------------------------#
#Seleccion de estaciones
estaciones = pd.read_csv(data_dir + 'Estaciones_lista.csv')

aux = selecteras(estaciones, -63, -55) #en longitud
est_sel = selecteras(aux, -40, -30, n_col=1) #en longitud
#-----------------------------------------------------------------------------#
#Seleccion de datos
est_centro = pd.read_csv(data_dir + 'ncar-centro.txt', sep='\s+', header=None)
est_norte = pd.read_csv(data_dir + 'ncar-norte.txt', sep='\s+', header=None)
est_pat = pd.read_csv(data_dir + 'ncar-patag.txt', sep='\s+', header=None)

aux = est_sel['OMM']*10
for e, i in zip(range(0, len(aux)), range(0, len(aux))):
    est_num = aux.iloc[e]
    aux_serie = est_centro.loc[(est_centro[0]==est_num)]
    if len(aux_serie)==0:
        aux_serie = est_norte.loc[(est_norte[0] == est_num)]
    if len(aux_serie) == 0:
        aux_serie = est_pat.loc[(est_pat[0] == est_num)]

    aux_ds = xr.Dataset(
        data_vars={
            'anio':(aux_serie[1]),
            # sacando los datos faltantes por np.nan
            'temp':(aux_serie[3].replace(990,np.nan)/10),
            'prec':(aux_serie[4].replace(200000,np.nan)/10)
        },
        coords={'estacion':est_num},
    )

    if i!=0:
        ds_f = xr.concat([ds_f, aux_ds],dim='estacion')
    else:
        ds_f = aux_ds
#-----------------------------------------------------------------------------#
#Control de calidad:
