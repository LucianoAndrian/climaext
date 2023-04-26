#-----------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import pymannkendall as mk
#-----------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p1/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p1/'
save = True
dpi = 200
plot_scatter_estaciones_sel = False
#-----------------------------------------------------------------------------#
def SelectAreas(df, low, top, n_col=0):
    return df.loc[(df[df.columns[n_col]]>=low)&(df[df.columns[n_col]]<=top)]

def SelEstacion(ds, num_est):
    return ds.where(ds.estacion==num_est, drop=True)

def is_months(month, mmin, mmax):
    return (month >= mmin) & (month <= mmax)
#-----------------------------------------------------------------------------#
# Region: PAMPA 30-40S 63-55W
#-----------------------------------------------------------------------------#
#Seleccion de estaciones
estaciones = pd.read_csv(data_dir + 'Estaciones_lista.csv')

aux = SelectAreas(estaciones, -63, -55) #en longitud
est_sel = SelectAreas(aux, -40, -30, n_col=1) #en longitud
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
df_30_11 = df_30.loc[(df['end']-df['start']>30) & (df['maxtimestep']<=5)]

# -----------------------------------------------------------------------------#
# ploteo de las estaciones para ambas variables
if plot_scatter_estaciones_sel:
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
            ax.set_title(str(df_30_11['estacion'][n]) + ' - ' + str(n))
            if save:
                plt.savefig(out_dir + str(df_30_11['estacion'][n]) +
                            '_' + str(n) + '.jpg')
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
1970-2003 x12
1961-2003 x11
1951-2003 x10
1931-2003 x6

Prec.
1970-2003 x12
1961-2003 x11
1951-2003 x10
1931-2003 x6
1888-2003 x3
"""
# -----------------------------------------------------------------------------#
# Calculando todo para los distintos periodos
# -----------------------------------------------------------------------------#

est_sel['OMM'] = est_sel['OMM']*10

for start in [1970, 1961,1951, 1934]:
    # Seleccion del periodo a partir del criterio aplicado
    df_sel = df_30_11.loc[(df_30_11['end'] == 2003) &
                          (df_30_11['start'] <= start)]
    est_sel_check = len(df_sel)
    years_check = 2003-start+1

    # Seleccion de los datos de las estaciones
    i = 0
    for est_n in range(0, 23):
        try:
            aux_nombre = df_sel['estacion'][est_n]
            aux_ds = SelEstacion(ds_f, aux_nombre)
            aux_ds = aux_ds.sel(time=slice(start, 2003))

            # en caso que falte algun anio
            if len(np.unique(aux_ds.time.values)) < est_sel_check:
                print('Error en start: ' + str(start))
                break

            # reemplazando con dates para facilitar
            # la seleccion futura
            time = pd.date_range(start=str(start-1)+ '-12-01',
                                 end='2003-12-01', freq='M') \
                   + pd.DateOffset(days=1)
            aux_ds['time'] = time

            if i != 0:
                ds_sel = xr.concat([ds_sel, aux_ds], dim='time')
            else:
                ds_sel = aux_ds
                i += 1
        except:
            pass

    # -------------------------------------------------------------------------#
    # Calculos
    print('Calculando...')
    # -------------------------------------------------------------------------#
    i = 0
    i2 = 0
    for est_n in range(0, 23):
        try:
            aux_nombre = df_sel['estacion'][est_n]
            aux = SelEstacion(ds_sel, aux_nombre)
            # ------------------------------------------------------------------#
            # Promedio anual T y acumulado PP
            aux_pp = aux.prec.groupby('time.year').sum()
            aux_t = aux.temp.groupby('time.year').mean()
            # ------------------------------------------------------------------#
            # marchas anuales
            aux_marchas = aux.groupby('time.month').mean()
            # ------------------------------------------------------------------#
            # promedios y acumulados mensuales
            aux_mam = aux.sel(time=is_months(aux['time.month'], mmin=3, mmax=5))
            aux_mam_pp = aux_mam.prec.groupby('time.year').sum()
            aux_mam_t = aux_mam.temp.groupby('time.year').mean()

            aux_jja = aux.sel(time=is_months(aux['time.month'], mmin=6, mmax=8))
            aux_jja_pp = aux_jja.prec.groupby('time.year').sum()
            aux_jja_t = aux_jja.temp.groupby('time.year').mean()

            aux_son = aux.sel(time=is_months(aux['time.month'], mmin=9, mmax=11))
            aux_son_pp = aux_son.prec.groupby('time.year').sum()
            aux_son_t = aux_son.temp.groupby('time.year').mean()

            # djf!
            aux_jf = aux.sel(time=aux.time.dt.month.isin([1, 2]))
            # Selecciono diciembre y le cambio la fecha para pasarlo al proximo año
            aux_d = aux.sel(time=aux.time.dt.month.isin([12]))
            aux_d['time'] = pd.date_range(start=str(start+1) + '-12-01',
                                          end='2005-12-01', freq='A-DEC')
            # juntando d con jf
            aux_djf = xr.concat([aux_jf, aux_d], dim='time')
            aux_djf_pp = aux_djf.prec.groupby('time.year').sum()[:-1]
            aux_djf_t = aux_djf.temp.groupby('time.year').mean()[:-1]
            # ------------------------------------------------------------------#
            d = {'estacion': [aux_nombre] * len(aux_t.year.values),
                 'anios': aux_t.year.values,
                 'tmeany': aux_t.values, 'ppacumy': aux_pp.values,
                 'tmam': aux_mam_t.values, 'ppmam': aux_mam_pp.values,
                 'tjja': aux_jja_t.values, 'ppjja': aux_jja_pp.values,
                 'tson': aux_son_t.values, 'ppson': aux_son_pp.values,
                 'tdjf': aux_djf_t.values, 'ppdjf': aux_djf_pp.values}

            if i == 0:
                i += 1
                df = pd.DataFrame(d)
            else:
                df = df.append(pd.DataFrame(d), ignore_index=True)

            # ------------------------------------------------------------------#
            d2 = {'estacion': [aux_nombre] * 12, 'anios': np.linspace(1, 12, 12),
                  'tmarcha': aux_marchas.temp.values,
                  'ppmarcha': aux_marchas.prec.values}

            if i2 == 0:
                i2 += 1
                df_marchas = pd.DataFrame(d2)
            else:
                df_marchas = df_marchas.append(pd.DataFrame(d2), ignore_index=True)
        except:
            pass

    # Test: se computaros todas las estaciones seleccionadas?
    if est_sel_check == (len(df))/years_check:
        print('start: ' + str(start) + ' OK')
        # guardado, stand by...
        # HAY Q SACAR LOS NANs...
        # np.savetxt(out_dir + str(start) + '_2003.txt', df, fmt='%d',
        #            delimiter='\t')
        # np.savetxt(out_dir + str(start) + '_2003_marchas.txt', df_marchas,
        #            fmt='%d', delimiter='\t')

    print('Calculo y testeo de tendencia')
    for aux_nombre in np.unique(df.estacion):
        df_aux = df.loc[(df['estacion'] == aux_nombre)]
        df_aux_marchas = df_marchas.loc[(df_marchas['estacion'] == aux_nombre)]
        df_trends_aux = pd.DataFrame({'estacion': [str(aux_nombre)]})
        for col in df_aux.columns[2:]:
            mk_test = mk.original_test(df_aux[col])

            d = pd.DataFrame({col: [np.round(mk_test.slope, 2)],
                              col + 'sig': [mk_test.h]})

            df_trends_aux = pd.concat([df_trends_aux, d], axis=1)

        # Seleccionar lat y lon de las estaciones
        aux_coords = est_sel.loc[(est_sel['OMM'] == aux_nombre)]
        d2 = pd.DataFrame({'nombre': aux_coords.Nombre.values[0],
                           'lon': [aux_coords['Lon'].values[0]],
                           'lat': [aux_coords['Lat'].values[0]]})

        # Ploteos Series
        print('Ploteando series...')
        # Temperatura -------------------------------------------------------------#
        fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
        ax = fig.add_subplot(111)

        tmeany = ax.plot(df_aux.anios, df_aux.tmeany, '-o', label='Anual', color='k'
                         , linewidth=2,
                         markersize=.5, markerfacecolor='k', markeredgecolor='white')
        tmam = ax.plot(df_aux.anios, df_aux.tmam, '-o', label='MAM', color='gold',
                       linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='orange')
        tjja = ax.plot(df_aux.anios, df_aux.tjja, '-o', label='JJA', color='dodgerblue',
                       linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='turquoise')
        tson = ax.plot(df_aux.anios, df_aux.tson, '-o', label='SON', color='green',
                       linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='lime')
        tdjf = ax.plot(df_aux.anios, df_aux.tdjf, '-o', label='DJF', color='firebrick'
                       , linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='red')
        ax.set_ylim(5, 30)
        ax.set_ylabel('[ºC]')
        ax.set_xlabel('Años')
        lns = tmeany + tmam + tjja + tson + tdjf
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        ax.grid()
        ax.set_title(d2.nombre.values[0] + ' - Temperatura - ' + str(start) + '-2003')
        if save:
            plt.savefig(out_dir + str(aux_nombre) + '_series_Temp_' + str(start) + '-2003.jpg')
            print('Save')
            plt.close('all')
        else:
            plt.show()

        # Precipitacion - Estacional  # -----------------------------------------------#
        fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
        ax = fig.add_subplot(111)
        tmam = ax.plot(df_aux.anios, df_aux.ppmam, '-o', label='MAM', color='gold',
                       linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='orange')
        tjja = ax.plot(df_aux.anios, df_aux.ppjja, '-o', label='JJA', color='dodgerblue',
                       linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='turquoise')
        tson = ax.plot(df_aux.anios, df_aux.ppson, '-o', label='SON', color='green',
                       linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='lime')
        tdjf = ax.plot(df_aux.anios, df_aux.ppdjf, '-o', label='DJF', color='firebrick'
                       , linewidth=2,
                       markersize=.5, markerfacecolor='k', markeredgecolor='red')
        ax.set_ylim(0, 700)
        ax.set_ylabel('[mm]')
        ax.set_xlabel('Años')
        lns = tmam + tjja + tson + tdjf
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        ax.grid()
        ax.set_title(d2.nombre.values[0] + ' - Precipitacion - ' + str(start) + '-2003')
        if save:
            plt.savefig(out_dir +  str(aux_nombre)+ '_series_Prec_' + str(start) + '-2003.jpg')
            print('Save')
            plt.close('all')
        else:
            plt.show()

        # Precipitacion - ANUAL -------------------------------------------------------#
        fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
        ax = fig.add_subplot(111)
        tmeany = ax.plot(df_aux.anios, df_aux.ppacumy, '-o', label='Anual', color='k'
                         , linewidth=2,
                         markersize=.5, markerfacecolor='k', markeredgecolor='white')
        ax.set_ylim(100, 1700)
        # ax2.set_ylim(400,1500)
        ax.set_ylabel('[mm] Anual')
        ax.set_xlabel('Años')
        lns = tmam + tjja + tson + tdjf
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left')
        ax.grid()
        ax.set_title(d2.nombre.values[0] + ' - Precipitacion - ' + str(start) + '-2003')
        if save:
            plt.savefig(out_dir +  str(aux_nombre) + '_series_Prec_' + str(start) + '-2003.jpg')
            print('Save')
            plt.close('all')
        else:
            plt.show()
        # -----------------------------------------------------------------------------#
        # Marchas
        fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()
        tmarcha = ax2.plot(df_aux_marchas.anios, df_aux_marchas.tmarcha,
                          label='Temp', color='firebrick'
                         , linewidth=4, markersize=.5, markerfacecolor='k',
                          markeredgecolor='red')
        ppmarcha = ax.bar(df_aux_marchas.anios, df_aux_marchas.ppmarcha,
                          label='Prec')
        ax.set_ylim(0, 200)
        ax2.set_ylim(5,25)
        ax.set_ylabel('[mm]')
        ax2.set_ylabel('[ºC]')
        ax.set_xlabel('Mes')
        ax.grid()
        ax.set_title(d2.nombre.values[0] + ' - Marchas Anuales - ' + str(start) + '-2003')
        if save:
            plt.savefig(out_dir + str(aux_nombre) + '_marchas_' + str(start) + '-2003.jpg')
            print('Save')
            plt.close('all')
        else:
            plt.show()
        # -----------------------------------------------------------------------------#
        # -----------------------------------------------------------------------------#

        if aux_nombre == np.unique(df.estacion)[0]:

            df_trends = pd.concat([d2, df_trends_aux], axis=1)
        else:

            df_trends_aux = pd.concat([d2, df_trends_aux], axis=1)
            df_trends = df_trends.append(df_trends_aux, ignore_index=True)

    print('Save df_trends in ' + str(start) + '-2003')
    np.savetxt(out_dir + 'df_trends_' + str(start) + '_2003.txt', df_trends, fmt='%s', delimiter='\t')
# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#
# Tendencia y significancia
# para no asumir ninguna distribucion
# Mann-Kendall, no parametrico
# -----------------------------------------------------------------------------#

# incorporar a lo anteriod
# argegarle titulo,. lat lon, periodo
# guardar

# mapas
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from cartopy.io.img_tiles import StamenTerrain


fig = plt.figure(figsize=(5, 6), dpi=100)
crs_latlon = ccrs.PlateCarree()
tiler = StamenTerrain()
mercator = tiler.crs
ax = fig.add_subplot(111, projection=ccrs.PlateCarree(central_longitude=0))
ax.add_image(tiler, 8)
plt.scatter(df_trends.lon.values, df_trends.lat.values,
           s=df_trends.tson.values*10000,
           c = df_trends.tson.values,
            cmap='viridis')

# cuales son significativas
df_coords_aux = df_trends.loc[df_trends['tsonsig']==True]

# ax.scatter(df_coords_aux.lon.values, df_coords_aux.lat.values,
#            c=df_coords_aux.tsonsig.values, edgecolor='k',
#            s=200, marker= 'x')
ax.scatter(df_coords_aux.lon.values, df_coords_aux.lat.values,
           c='red', edgecolor='red',
           s=100, marker= 'P')
plt.colorbar(fraction=0.042, pad=0.035,shrink=0.7)
#ax.stock_img()
ax.coastlines('10m')
ax.set_extent([-63, -55, -40, -30], crs_latlon)
ax.add_feature(cartopy.feature.LAND, facecolor='white')
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.COASTLINE)
ax.add_feature(cartopy.feature.STATES)
plt.tight_layout()
plt.show()


