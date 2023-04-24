#-----------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
#-----------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p1/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p1/'
save = True
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
est_sel_num = []
for e, i in zip(range(0, len(aux)), range(0, len(aux))):
    est_num = aux.iloc[e]
    aux_serie = est_centro.loc[(est_centro[0]==est_num)]
    if len(aux_serie)==0:
        aux_serie = est_norte.loc[(est_norte[0] == est_num)]
    if len(aux_serie) == 0:
        aux_serie = est_pat.loc[(est_pat[0] == est_num)]
    #
    # aux_ds = xr.Dataset(
    #     data_vars={
    #         'anio':(aux_serie[1].values),
    #         # sacando los datos faltantes por np.nan
    #         'temp':(aux_serie[3].replace(990,np.nan)/10).values,
    #         'prec':(aux_serie[4].replace(200000,np.nan)/10).values
    #     },
    #     coords={'estacion':est_num},
    # )
    aux_serie_pp = aux_serie[4].replace(200000,np.nan)/10
    aux_serie_pp = aux_serie_pp.replace(15000,0.05) #traza
    aux_ds = xr.Dataset(
        data_vars=dict(
            temp=(['time'], (aux_serie[3].replace(990,np.nan)/10).values),
            prec=(['time'], aux_serie_pp.values),
        ),
        coords=dict(
            time=aux_serie[1].values,
            estacion=est_num
        ),
    )

    est_sel_num.append(est_num)

    if i!=0:
        ds_f = xr.concat([ds_f, aux_ds],dim='time')
    else:
        ds_f = aux_ds

def SelEstacion(ds, num_est):
    return ds.where(ds.estacion==num_est, drop=True)
#-----------------------------------------------------------------------------#
# las estaciones tienen distintos piodos observados
# cual es el periodo en común mas largo?
# que porcentaje de datos faltantes en total tiene cada una?
#-----------------------------------------------------------------------------#
for n in est_sel_num:
    aux = SelEstacion(ds_f, n)
    temp_nan = len(aux.temp[aux.temp.notnull()])/len(aux.temp)
    pp_nan = len(aux.temp[aux.prec.notnull()])/len(aux.prec)
    check_time_step = []
    for n2, i in zip(aux.time.values, range(0, len(aux.time.values))):
        try:
            check_time_step.append(aux.time.values[i+1]-n2)
        except:
            pass

    nan_count_temp = 0
    nan_count_pp = 0
    for n3 in np.unique(aux.time.values):
         if len(aux.sel(time=n3).temp.values[np.isnan(
                 aux.sel(time=n3).temp.values)]) > 5:
             nan_count_temp += 1

         if len(aux.sel(time=n3).prec.values[np.isnan(
                 aux.sel(time=n3).prec.values)]) > 5:
             nan_count_pp += 1

# -----------------------------------------------------------------------------#
    d = {'estacion': [n], 'start': [aux.time[0].values],
         'end': [aux.time[-1].values], 'tnan': [np.round(1-temp_nan,2)*100],
         'ppnan': [np.round(1-pp_nan,2)*100],
         'maxtimestep': [np.max(check_time_step)],
         'hynan_temp': [nan_count_temp],
         'hynan_prec': [nan_count_pp]}

    if n == est_sel_num[0]:
        df = pd.DataFrame(d)
    else:
        df = df.append(pd.DataFrame(d), ignore_index=True)


np.savetxt(out_dir + 'datos_est_sel.txt', df, fmt='%d')

# -----------------------------------------------------------------------------#
# Total 22 estaciones
# Sin tener en cuenta los faltantes, solo periodos de mas de 30 años
# quedan 15 estaciones
df_30 = df.loc[(df['end']-df['start']>=30)]

# solo periodos de mas de 30 que tengan un periodo de menos de 11 años salteados
# (quedan de 11 años pero en estaciones de mas de 100 años. se pueden separar
# los periodos de estudio)
# quedan 15 estaciones
df_30_11 = df_30.loc[(df['end']-df['start']>30) & (df['maxtimestep']<=11)]

# -----------------------------------------------------------------------------#
# ploteo de las estaciones para ambas variables
for n in range(0, 22):
    try:
        aux = SelEstacion(ds_f, df_30_11['estacion'][n])
        fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        temp_tl = ax.scatter(aux.time.values, aux.temp.values,
                             color='firebrick', label='Temp')
        prec_tl = ax2.scatter(aux.time.values, aux.prec.values,
                              color='dodgerblue', label='Prec')

        ax.set_ylim(5, 30)
        ax.set_ylabel('Temp. [ºC]')
        ax2.set_ylim(0, 500)
        ax2.set_ylabel('Prec. [mm]')
        ax.set_xlabel('Años')

        lns = [temp_tl] + [prec_tl]
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        ax.set_title(str(df_30_11['estacion'][n]) + ' - ' +  str(n))
        if save:
            plt.savefig(out_dir + str(df_30_11['estacion'][n]) +
                        '_' +  str(n) + '.jpg')
            print('Save')
            plt.close('all')
        else:
            plt.show()
    except:
        pass



"""
---------------------------------------0: 1931-2003 
--------------------------2: Temp 1951-2003, pp: 1900-1940, 1951-2003
--------------------------3: 1941-2003
--------------4: 1961-2003
--------------------------6: 1951-2003
---------------------------------------8: ~1931 - 2003
--------------------------9: 1951-2003
--------------------------10: 1860-2003
13: 1931-1994-97 
---------------------------------------14: 1931-2003
-------------16: 1931-1960 1970-2003
---------------------------------------17: Temp 1931-2003 pp: 1888-2003
---------------------------------------21: temp 1860-1880 1890-2003 pp: 1860-2003

Periodo mas corto: 1961-1994 debido a una estacion. limita mucho --> chau! chau que? estoy saludando
(chau las dos, la que empieza en 1961 y la que termina en 1994)



El CRITERIO:
Periodos mayores a 30 años con periodos faltantes de < 3años
Considerando los periodos mas largos que se pueden tomar y teniendo en cuenta que todas las
estaciones presentan nans se descartan la 13 y la 16. La 16 se podria incluir en un periodo de 33 años comun a todas
pero el mas corto

Con un total de 11 estaciones de 22 inciales se pueden estudiar los siquientes periodos:
Periodos mas largos:
Temp. 
1961-2003 x11
1951-2003 x10
1931-2003 x6

Prec.
1961-2003 x11
1951-2003 x10
1931-2003 x6
1888-2003 x3
"""


















