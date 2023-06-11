# practica 3
################################################################################
# sólo ploteos
################################################################################
import xarray as xr
import numpy as np
from numpy import ma
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings("ignore")
################################################################################
data_dir = '/home/luciano.andrian/doc/climaext/p3/ncfiles/'
out_dir = '/home/luciano.andrian/doc/climaext/p3/salidas/'
data_dir2 = '/home/luciano.andrian/doc/climaext/p2/data/'
data_dir3 = '/home/luciano.andrian/doc/climaext/p3/ncfiles2/'
data_dir4 = '/home/luciano.andrian/doc/climaext/p3/ncfiles_km/'

save = False
dpi = 50
################################################################################
def SelectFiles(dir):
    files =  glob.glob(dir +'*.nc')
    return sorted(files, key=lambda x: x.split()[0])

def FindMonth(f):
    try:
        int(int(f.split('.')[-2].split('_')[-1]))
        return f.split('_')[-2]
    except:
        return f.split('_')[-3]

def FindYear(f):
    try:
        int(int(f.split('.')[-2].split('_')[-1]))
        return f.split('.')[-2].split('_')[-1]
    except:
        return f.split('.')[-2].split('_')[-2]

def Plot(comp, comp_var, levels, save, dpi, title, name_fig, out_dir,
         color_map, cmap, contourf, sa):

    import matplotlib.pyplot as plt
    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs
    if sa:
        fig_size = (5, 6)
        extent = [270, 330, -60, 20]
        xticks = np.arange(270, 330, 10)
        yticks = np.arange(-60, 40, 20)
    else:
        fig_size = (6, 5)
        extent = [270, 320, -60, -20]
        xticks = np.arange(270, 320, 10)
        yticks = np.arange(-60, 0, 20)

    crs_latlon = ccrs.PlateCarree()


    levels_contour = levels.copy()
    if isinstance(levels, np.ndarray):
        levels_contour = levels[levels != 0]
    else:
        levels_contour.remove(0)


    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(extent, crs=crs_latlon)
    im = ax.contourf(comp.lon, comp.lat, comp_var, levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    if contourf:
        ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                    transform=crs_latlon, cmap=cmap, extend='both')

    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, zorder=17)
    # ax.add_feature(cartopy.feature.COASTLINE)
    if contourf:
        pass
    else:
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean',
                                                scale='50m', facecolor='white',
                                                alpha=1)
        ax.add_feature(ocean, linewidth=0.2, zorder=15)

    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', color=color_map)
    #ax.add_feature(cartopy.feature.RIVERS, edgecolor='skyblue')
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines(color=color_map, linestyle='-', alpha=1)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', zorder=20)
    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def PlotViento(comp, comp_var, px, py, levels, save, dpi, title, name_fig,
               out_dir, color_map, cmap, contourf, sa):

    import matplotlib.pyplot as plt
    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs

    if sa:
        fig_size = (5, 6)
        extent = [270, 330, -60, 20]
        xticks = np.arange(270, 330, 10)
        yticks = np.arange(-60, 40, 20)
    else:
        fig_size = (6, 5)
        extent = [270, 320, -60, -20]
        xticks = np.arange(270, 320, 10)
        yticks = np.arange(-60, 0, 20)

    crs_latlon = ccrs.PlateCarree()

    levels_contour = levels.copy()
    if isinstance(levels, np.ndarray):
        levels_contour = levels[levels != 0]
    else:
        levels_contour.remove(0)

    fig = plt.figure(figsize=fig_size, dpi=dpi)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent(extent, crs=crs_latlon)
    im = ax.contourf(comp.lon, comp.lat, comp_var, levels=levels,
                     transform=crs_latlon, cmap=cmap, extend='both')

    cb = plt.colorbar(im, fraction=0.042, pad=0.035, shrink=0.8)
    if contourf:
        ax.contour(comp.lon, comp.lat, comp_var, levels=levels,
                   transform=crs_latlon, cmap=cmap, extend='both')

    cb.ax.tick_params(labelsize=8)

    Q60 = np.percentile(np.sqrt(np.add(np.power(px, 2), np.power(py, 2))), 0)
    M = np.sqrt(np.add(np.power(px, 2), np.power(py, 2))) < Q60
    # mask array
    px_mask = ma.array(px, mask=M)
    py_mask = ma.array(py, mask=M)
    # plot vectors
    lons, lats = np.meshgrid(comp.lon.values, comp.lat.values)
    ax.quiver(lons, lats, px_mask, py_mask, transform=crs_latlon,
              pivot='tail',
              width=0.0030, headwidth=4.1, alpha=1, color='k',
              scale=40)
    ax.add_feature(cartopy.feature.LAND, facecolor='white',
                   edgecolor=color_map)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, zorder=17)
    # ax.add_feature(cartopy.feature.COASTLINE)
    if contourf:
        pass
    else:
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean',
                                                    scale='50m',
                                                    facecolor='white',
                                                    alpha=1)
        ax.add_feature(ocean, linewidth=0.2, zorder=15)
    ax.add_feature(cartopy.feature.BORDERS, linestyle='-', color=color_map)
    # ax.add_feature(cartopy.feature.RIVERS, edgecolor='skyblue')
    ax.add_feature(cartopy.feature.STATES)
    ax.coastlines(color=color_map, linestyle='-', alpha=1)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-')
    ax.set_xticks(xticks, crs=crs_latlon)
    ax.set_yticks(yticks, crs=crs_latlon)
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.gridlines(crs=crs_latlon, linewidth=0.3, linestyle='-', zorder=20)
    ax.tick_params(labelsize=7)
    plt.title(title, fontsize=10)
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

cbar_pp = colors.ListedColormap(['#003C30', '#004C42', '#0C7169',
                                 '#79C8BC', '#B4E2DB',
                                'white',
                                '#F1DFB3', '#DCBC75',
                                 '#995D13', '#6A3D07', '#543005', ][::-1])
cbar_pp.set_under('#3F2404')
cbar_pp.set_over('#00221A')
cbar_pp.set_bad(color='white')

cbar_t = colors.ListedColormap(['#B9391B', '#CD4838', '#E25E55',
                                '#F28C89', '#FFCECC',
                              'white',
                              '#B3DBFF', '#83B9EB', '#5E9AD7',
                                '#3C7DC3', '#2064AF'][::-1])
cbar_t.set_over('#9B1C00')
cbar_t.set_under('#014A9B')
cbar_t.set_bad(color='white')

################################################################################
files = SelectFiles(data_dir)
# pp t y hgt850
for f in files:
    data = xr.open_dataset(f)
    variable = f.split('_')[0].split('/')[-1]
    if variable == 'gpcc' or variable == 'pp':
        try:
            int(f.split('_')[-1].split('.')[0])
            # no anom
            title = 'PP - ' + FindMonth(f) + '_' + FindYear(
                f) + ' - ' + variable
            name_fig = 'pp_' + FindMonth(f) + '_' + FindYear(f) + '_' + variable
            scale = np.arange(0, 450, 50)
            cbar = 'YlGnBu'
        except:
            title = 'PP anom. - ' + FindMonth(f) + '_' + FindYear(f) + ' - ' \
                    + variable
            name_fig = 'pp_anom_' + FindMonth(f) + '_' + FindYear(f) + '_' +\
                       variable
            scale = [-100, -75, -50, -25, -5, 0, 5, 25, 50, 75, 100]
            cbar = cbar_pp

        if variable == 'pp':
            data = data.rename({'prate': 'precip'})
            data *= (365 / 12)

        Plot(data, data.precip[0, :, :], scale, save, dpi,
             title, name_fig, out_dir, 'gray', cbar, False, False)

    elif variable == 't':
        try:
            int(f.split('_')[-1].split('.')[0])
            # no anom
            title = 'Temp - ' + FindMonth(f) + '_' + FindYear(f) \
                    + ' - ' + variable
            name_fig = 't_' + f.split('_')[1] + '_' + FindMonth(f) + '_' + \
                       FindYear(f) + '_' + variable
            scale = np.arange(-5, 40, 5)
            cbar = 'Spectral_r'
        except:
            title = 'Temp anom. - ' + FindMonth(f) + '_' + FindYear(f)
            name_fig = 't_anom_' + f.split('_')[1] + '_' + FindMonth(f) + '_' +\
                       FindYear(f)
            scale = [-5,-2,-1,-.5,0,.5,1,2,5]
            cbar = cbar_t
        Plot(data, data.air[0, :, :], scale, save, dpi,
             title, name_fig, out_dir, 'gray', cbar, False, False)
    elif 'hgt' in variable:
        try:
            int(f.split('_')[-1].split('.')[0])
            # no anom
            title = 'Temp - ' + FindMonth(f) + '_' + FindYear(f) \
                    + ' - ' + variable
            name_fig = 't_' + f.split('_')[1] + '_' + FindMonth(f) + '_' + \
                       FindYear(f) + '_' + variable
            scale = np.arange(1200, 1800, 50)
            cbar = 'Reds'
        except:
            title = 'Temp anom. - ' + FindMonth(f) + '_' + FindYear(f)
            name_fig = 't_anom_' + f.split('_')[1] + '_' + FindMonth(f) + '_' + \
                       FindYear(f)
            scale = [-100, -75, -50, -15, -5, 0, 5, 15, 50, 75, 100]
            cbar = cbar_t

        Plot(data, data.hgt[0, :, :], scale, save, dpi,
             title, name_fig, out_dir, 'gray', cbar, True, False)

files_viento = files[-12::]
f = files_viento[1]

f_count=0
for f in files_viento:
    if ('u' in f.split('_')[0].split('/')[-1]):
        u = xr.open_dataset(f)

        checkf2 = True
        f_count2 = f_count
        while checkf2:
            f_count2 += 1
            try:
                f2 = files_viento[f_count2]
            except:
                f2 = files_viento[-1]
            if f.split('850_')[-1] in f2:
                checkf2 = False

        # este if está de demás pero por seguridad
        if ('v' in f2.split('_')[0].split('/')[-1]):
            v = xr.open_dataset(f2)

            auxu = u.rename({'uwnd': 'mag'})
            auxv = v.rename({'vwnd': 'mag'})
            mag = np.sqrt(auxu ** 2 + auxv ** 2)
            cbar = 'RdPu'

            try:
                int(f.split('_')[-1].split('.')[0])
                scale = np.arange(2, 20, 2)
            except:
                scale = np.arange(0, 10, 1)

                pass

            PlotViento(mag, mag.mag[0, :, :], u.uwnd[0, :, :], v.vwnd[0, :, :],
                      scale,
                     save, dpi, 'title', 'name_fig', out_dir, 'gray', cbar, True,
                     False)
            print(f)
            print(f2)
    f_count += 1

################################################################################
################################################################################
# 2
################################################################################
# reciclando de p2...
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

def checkEx(out, media, mes, smnexmax, smnexmin, variable):
    outvar = out[variable].values[0]

    d = {'estacion': [0], 'dia': [0], 'mes': [0], 'anio': [0],
         'tx': [0], 'tm': [0], 'pp': [0], 'ppc': [0], 'h': [0],
         'dv': [0], 'vx': [0]}
    d = pd.DataFrame(d)

    if outvar > media:
        if outvar > smnexmax[mes-1]:
            return out
        else:
            return []
    elif outvar < media:
        if outvar < smnexmin[mes-1]:
            return out
        else:
            return []

def PeriodosTemp(df, perc_t):
    def checkEx2(data):
        try:
            len(data)
            return data
        except:
            d = {'estacion': [0], 'dia': [0], 'mes': [0], 'anio': [0],
                 'tx': [0], 'tm': [0], 'pp': [0], 'ppc': [0], 'h': [0],
                 'dv': [0], 'vx': [0]}
            d = pd.DataFrame(d)
            return d


    tm10_count = 0
    tm01_count = 0
    tx90_count = 0
    tx99_count = 0
    tm90_count = 0
    tm99_count = 0
    for mes in range(1, 13):
        mc = None
        ec = None
        mf = None
        ef = None
        for dia in range(1, 32):
            if (mes == 2) and (dia == 29):
                pass
            else:
                dia_aux = df.loc[(df['mes'] == mes) & (df['dia'] == dia)]

                if len(dia_aux) == 0:
                    pass
                else:
                    aux = perc_t.loc[
                        (perc_t['dia'] == dia) & (perc_t['mes'] == mes)]

                    tx90 = aux.tx_9.values[0]
                    tx99 = aux.tx_99.values[0]
                    tm90 = aux.tm_9.values[0]
                    tm99 = aux.tm_99.values[0]
                    tm10 = aux.tm_1.values[0]
                    tm01 = aux.tm_01.values[0]

                    # muy calidos
                    if dia_aux.tx.values[0] > tx90:
                        if tx90_count == 0:
                            mc = dia_aux
                            tx90_count = 1
                        else:
                            mc = pd.concat([mc, dia_aux], axis=0)

                        # extremadamente calidos
                        if dia_aux.tx.values[0] > tx99:
                            if tx99_count == 0:
                                ec = dia_aux
                                tx99_count = 1
                            else:
                                ec = pd.concat([ec, dia_aux], axis=0)

                    # muy calidos desde tm
                    if dia_aux.tm.values[0] > tm90:
                        if tm90_count == 0:
                            mc_tm = dia_aux
                            tm90_count = 1
                        else:
                            mc_tm = pd.concat([mc_tm, dia_aux], axis=0)

                        # extremadamente calidos
                        if dia_aux.tm.values[0] > tm99:
                            if tm99_count == 0:
                                ec_tm = dia_aux
                                tm99_count = 1
                            else:
                                ec_tm = pd.concat([ec_tm, dia_aux], axis=0)

                    # muy frios
                    if dia_aux.tm.values[0] < tm10:
                        if tm10_count == 0:
                            mf = dia_aux
                            tm10_count = 1
                        else:
                            mf = pd.concat([mf, dia_aux], axis=0)

                        # extremadamente frio
                        if dia_aux.tm.values[0] < tm01:
                            if tm01_count == 0:
                                ef = dia_aux
                                tm01_count = 1
                            else:
                                ef = pd.concat([ef, dia_aux], axis=0)
        if mes == 1:
            meses_mc = checkEx2(mc)
            meses_ec = checkEx2(ec)
            meses_mc_tm = checkEx2(mc_tm)
            meses_ec_tm = checkEx2(ec_tm)
            meses_mf = checkEx2(mf)
            meses_ef = checkEx2(ef)
        else:
            meses_mc = pd.concat([meses_mc, checkEx2(mc)], axis=0)
            meses_ec = pd.concat([meses_ec, checkEx2(ec)], axis=0)
            meses_mc_tm = pd.concat([meses_mc_tm, checkEx2(mc_tm)], axis=0)
            meses_ec_tm = pd.concat([meses_ec_tm, checkEx2(ec_tm)], axis=0)
            meses_mf = pd.concat([meses_mf, checkEx2(mf)], axis=0)
            meses_ef = pd.concat([meses_ef, checkEx2(ef)], axis=0)

    return meses_mc, meses_ec, meses_mc_tm, meses_ec_tm, meses_mf, meses_ef

def DetecPeriodsx3(df):
    from itertools import groupby
    from operator import itemgetter
    check = True
    check_first = False
    for mes in range(1, 13):
        aux = df.loc[df['mes'] == mes]
        per_f = []
        aux_mes = []
        if len(aux) >= 3:

            for k, g in groupby(enumerate(aux['dia'].values),
                                lambda ix: ix[0] - ix[1]):
                aux_mes.append(list(map(itemgetter(1), g)))
            periodos = [len(l) >= 3 for l in aux_mes]

            if True in periodos:
                check_first = True
                for l in range(0, len(periodos)):
                    if periodos[l]:
                        per_f.append(aux_mes[l])
                # ESTO PUEDE Q NO HAGA FALTA
                for l in range(0, len(per_f)):
                    if l == 0:
                        mes_per = aux.loc[aux['dia'].isin(per_f[l])]
                    else:
                        mes_per = pd.concat([mes_per,
                                             aux.loc[
                                                 aux['dia'].isin(per_f[l])]],
                                            axis=0)
        try:
            len(mes_per)
            if check & check_first:
                per_meses = mes_per
                check = False
            else:
                per_meses = pd.concat([per_meses, mes_per], axis=0)

            del mes_per
        except:
            pass
    try:
        return per_meses
    except:
        return []

def plotperiods(anio, perc, titulo, name_fig, dpi, save,
                anioper_mc, anioper_ec, anioper_mc_tm, anioper_ec_tm,
                anioper_mf, anioper_ef, xlabel, ylabel):
    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = fig.add_subplot(111)

    plt.fill_between(x=perc['date'].values,
                     y1=perc['tx_99'].values, y2=perc['tm_01'].values,
                     color='#B5B5B5', alpha=1, label='p01-p99', linewidth=0.0)

    plt.fill_between(x=perc['date'].values,
                     y1=perc['tx_9'].values, y2=perc['tm_1'].values,
                     color='#F9EEEE', alpha=1, label='p10-p90', linewidth=0.0)

    ax.plot(anio['date'].values, anio['tx'].values, label='Tx', color='k',
            linewidth=1)
    ax.plot(anio['date'].values, anio['tm'].values, label='Tn',
            color='blue', linewidth=1)

    try:
        ax.scatter(anioper_mc['date'].values, anioper_mc['tx'].values,
                   label='Muy Calido',
                   color='magenta', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_ec['date'].values, anioper_ec['tx'].values,
                   label='Ext. Calido',
                   color='red', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_mc_tm['date'].values, anioper_mc_tm['tm'].values,
                   label='Muy Calido Tn',
                   color='coral', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_ec_tm['date'].values, anioper_ec_tm['tm'].values,
                   label='Ext. Calido Tn',
                   color='orange', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_mf['date'].values, anioper_mf['tm'].values,
                   label='Muy Frio',
                   color='k', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_ef['date'].values, anioper_ef['tm'].values,
                   label='Ext. Frio',
                   color='purple', linewidth=3)
    except:
        pass

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.set_title(titulo)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(out_dir + name_fig +'.jpg')
        plt.close('all')
    else:
        plt.show()

def plotItem3c(anio, titulo, dpi, save, per_pp, anioacum,
               xlabel, ylabel, name_fig):
    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = fig.add_subplot(111)

    ax.bar(anio.dj, anio.pp, label='pp diaria', color='dodgerblue', width=2)

    # percentiles 95 y 99
    diasmeses = np.cumsum([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

    for q, c in zip(['pp_95', 'pp_99'], ['orange', 'firebrick']):
        first = True
        for dm in range(0, 12):
            if first:
                first = False
                ax.hlines(y=per_pp[q].values[dm], xmin=0, xmax=diasmeses[dm],
                          label='p' + q.split('_')[1], color=c, linewidth=2)
            else:
                ax.hlines(y=per_pp[q].values[dm], xmin=diasmeses[dm - 1],
                          xmax=diasmeses[dm], color=c, linewidth=2)

    first = True
    for dm in range(0, 12):
        if first:
            first = False
            ax.hlines(y=anioacum.ppa.values[dm], xmin=0, xmax=diasmeses[dm],
                      label='media \n mensual', color='indigo', linewidth=2)
        else:
            ax.hlines(y=anioacum.ppa.values[dm], xmin=diasmeses[dm - 1],
                      xmax=diasmeses[dm], color='indigo', linewidth=2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.set_title(titulo)
    ax.set_ylabel(xlabel)
    ax.set_xlabel(ylabel)
    ax.grid(alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(out_dir + name_fig +'.jpg')
        print('Save')
        plt.close('all')
    else:
        plt.show()

def adddate(df2, anioaux=None):
    try:
        df = df2.copy()
        try:
            df.anio
        except:
            df['anio'] = anioaux

        df['date'] = pd.to_datetime(
            dict(year=df.anio, month=df.mes, day=df.dia))
        return df
    except:
        return []
#------------------------------------------------------------------------------#
data = pd.read_csv(data_dir2 + 'estaciones_1959_2020.csv', sep=',', header=None)
data_2021 = pd.read_csv(data_dir2 + 'estaciones_2021.csv', sep=',', header=None)
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
jn['tm']=jn['tm'].replace([213.3], 13.3)
# 2 ---------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
# Calcular los percentiles diarios (1, 10, 90, 99) de tx y tm y los
# percentiles mensuales (95, 99) de la pp (no nula) para el período 1981-2010.
jn_60_21 = jn.loc[(jn['anio']>=1981)&(jn['anio']<=2010)]

# temp --------------------------#
for q in [.01, .10, .90, .99]:
    for mes in range(1,13):
        for dia in range(1,32):
            if (mes==2) and (dia==29):
                pass
            else:
                per = jn_60_21.loc[(jn_60_21['dia'] == dia) &
                                   (jn_60_21['mes'] == mes)].quantile(q)

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

# pp --------------------------#
#percentiles
for q in [.95,.99]:
    for mes in range(1, 13):
        per = jn_60_21.loc[(jn_60_21['mes'] == mes) &
                           (jn_60_21['pp']>0.1)].quantile(q)

        m = {'mes':[mes], 'pp_'+ str(q).split('.')[1]:[per.pp]}
        if mes == 1:
            meses = pd.DataFrame(m)
        else:
            meses = pd.concat([meses, pd.DataFrame(m)], axis=0)

    if q == .95:
        per_pp = meses
    else:
        per_pp = pd.concat([per_pp, meses.pp_99], axis=1)


# -----------------------------------------------------------------------------#
def T_ext(mes, diasmes, extremo):
    aux_mes = jn.loc[jn['mes'] == mes]

    check_count = 0
    t_count = 0
    i_count = 0
    for i in range(1960,2022):
        aux = aux_mes.loc[jn['anio'] == i]

        for d in range(1, diasmes+1):
            try:
                # Dia d del mes m del año i
                aux_d = aux.loc[(aux['dia'] == d)]

                # percentil del dia de arriba
                aux_d_per = perc_t.loc[
                    (perc_t['mes'] == mes) & (perc_t['dia'] == d)]

                if extremo == 'calido':
                    check = (aux_d.tx.values[0] >= aux_d_per.tx_99[0]) & \
                            (aux_d.tm.values[0] >= aux_d_per.tm_99[0])
                elif extremo == 'frio':
                    check = (aux_d.tx.values[0] <= aux_d_per.tx_1[0]) & \
                            (aux_d.tm.values[0] <= aux_d_per.tm_01[0])
                else:
                    print('ERROR <extremo>')
                    return

                if check:
                    t_count += 1
                    check_count += 1
                    if i_count == 0:
                        t_ext = aux_d
                        i_count = 1
                    else:
                        t_ext = pd.concat([t_ext, aux_d])

            except:
                print('ERROR in dia: ' + str(d), ' año: ' + str(i))
                pass

    print('Dias extremos: ' + str(t_count))
    return t_ext

def PPext(meses):
    ppaux_count = 0
    i_count = 0
    for i in range(1960, 2022):
        aux = jn.loc[jn['anio'] == i]

        pp_c_m = 0
        for m in meses:

            pp_count = 0
            for d in range(1, 32):
                try:
                    aux_d = aux.loc[(aux['mes'] == m) & (aux['dia'] == d)]

                    aux_d_per = per_pp.loc[(per_pp['mes'] == m)]

                    ppaux = aux_d.pp.values[0] >= aux_d_per.pp_99[0]

                    if ppaux:
                        ppaux_count += 1
                        if pp_count == 0:
                            dpp = aux_d
                            pp_count = 1
                        else:
                            dpp = pd.concat([dpp, aux_d], axis=0)
                except:
                    pass

            try:
                len(dpp)
                if pp_c_m == 0:
                    dppm = dpp
                    pp_c_m = 1
                    del dpp
                else:
                    dppm = pd.concat([dppm, dpp], axis=0)
                    del dpp
            except:
                pass

        try:
            len(dppm)
            if i_count == 0:
                ppmes = dppm
                del dppm
                i_count = 1
            else:
                ppmes = pd.concat([ppmes, dppm], axis=0)
                del dppm

        except:
            print('Error en ' + str(d) + '/' + str(m) + '/' + str(i))
            pass

    print('Dias extremos: ' + str(ppaux_count))
    return ppmes

txene = T_ext(1,31, 'calido')
txene = txene.sort_values('tx', ascending=False).head(20)

tmjul = T_ext(7,31, 'frio')
tmjul = tmjul.sort_values('tm', ascending=True).head(20)

ppjj = PPext([6,7,8])
ppma = PPext([3,4])

np.savetxt(out_dir + 'txene.txt', txene, fmt='%s')
np.savetxt(out_dir + 'tmjul.txt', tmjul, fmt='%s')
np.savetxt(out_dir + 'ppjj.txt', ppjj, fmt='%s')
np.savetxt(out_dir + 'ppma.txt', ppma, fmt='%s')

################################################################################
# 3 Ploteos hgt y q
scales_hgt = []
scale_q = []

files = SelectFiles(data_dir3)
for f in files:
    data = xr.open_dataset(f)
    var_related = f.split('_')[-3].split('/')[-1]
    try:
        level = int(f.split('_')[-2]) # hgt

        name_fig = 'hgt_' + str(level) + '_' + f.split('_')[-1]\
                   + '_' + var_related
        cbar = cbar_t

        if 'anom' in f.split('_')[-1]:
            title = "Z' - " + str(level) + 'hPa'
            if level == 1000:
                scale = [-100, -75,-50, -25, -5, 0,
                         5, 25,50,  75,100]
            elif level == 850:
                scale = [-100, -75, -50, -15, -5,
                         0, 5, 15, 50, 75, 100]
            else:
                scale = [-200, -150, -100, -50, -10,
                         0, 10, 50, 100, 150, 200]
        else:
            title = "Z mean - " + str(level) + 'hPa'
            if level == 1000:
                scale = np.arange(-450,475,75)
                scale = [-450, -300, -200, -150,  -50,
                         0,   50,  150,  200,  300, 450]

            elif level == 850:
                scale = np.arange(1200,1800,50)
                cbar = 'Reds'

            else:
                scale = np.arange(5000,7000,200)
                cbar = 'Reds'

        Plot(data, data.hgt[0, :, :], scale, save, dpi,
             title, name_fig, out_dir, 'gray', cbar, True, True)

    except:

        if 'q' in f.split('_')[-2]:
            name_fig = 'hgt_' + str(level) + '_' + f.split('_')[-2] \
                       + '_' + var_related
            if 'anom' in f.split('_')[-1]:
                titulo = "Q' - 850 hPa"
                scale = [-0.004, -.003, -.002, -.001, -.0005, 0,
                         .0005, .001, .002, .003, .004]
                cbar = cbar_pp
            else:
                titulo = "Q mean - 850 hPa"
                scale = np.arange(0,.016,.002)
                cbar = 'YlGnBu'

            Plot(data, data.shum[0, :, :], scale, save, dpi,
                 titulo, name_fig, out_dir, 'gray', cbar, True, True)
################################################################################
# viento si hay tiempo.
################################################################################
# 4
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits
import os
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# funciones kmeans

def SilhScore(X, title='title', name_fig='fig', save=False):
    from sklearn.metrics import silhouette_score

    with threadpool_limits(limits=1):
        kmeans_per_k = [KMeans(n_clusters=k, n_init=100,
                               random_state=666).fit(X)
                        for k in range(1, 15)]

        silhouette_scores = [silhouette_score(X, model.labels_)
                             for model in kmeans_per_k[1:]]

    plt.figure(figsize=(8, 3))
    plt.plot(range(2, 15), silhouette_scores, "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("Silhouette score", fontsize=14)
    plt.title(title)

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

def SSEScore(X, title='title', name_fig='fig', save=False):
    sse = []
    with threadpool_limits(limits=1):
        kmeans_per_k = [KMeans(n_clusters=k, n_init=100,
                               random_state=666).fit(X)
                        for k in range(1, 15)]

        sse.append([model.inertia_ for model in kmeans_per_k[1:]])

    plt.figure(figsize=(8, 3))
    plt.plot(range(2, 15), sse[0], "bo-")
    plt.xlabel("$k$", fontsize=14)
    plt.ylabel("SSE", fontsize=14)
    plt.title(title)

    if save:
        plt.savefig(out_dir + name_fig + '.jpg')
        plt.close()
    else:
        plt.show()

    return sse

def KM(data, n_clusters):
    with threadpool_limits(limits=1):  # con 1 mas rapido q con 40...
        kmeans = KMeans(
            n_clusters=n_clusters, n_init=100, random_state=666).fit(data)

        return kmeans.predict(X), kmeans.cluster_centers_

for a, m, nc in zip([2020, 2021], [3,4], [8,6]):

    aux = xr.open_dataset(data_dir4 + 'hgt.' + str(a) + '.nc')

    daily = aux.sel(time=aux.time.dt.month.isin(m), level=850,
                    lon=slice(270, 320), lat=slice(-20, -60))
    daily = daily.drop(['time_bnds', 'level'])

    aux = xr.open_dataset(data_dir4 + 'hgt.mon.ltm.1991-2020.nc')
    mmonth = aux.sel(time=aux.time.dt.month.isin(m), level=850,
                     lon=slice(270, 320), lat=slice(-20, -60))
    mmonth = mmonth.drop(
        ['valid_yr_count', 'climatology_bounds', 'level', 'time'])

    # Anomalia en funcion del mes
    anom = daily - mmonth.hgt[0, :, :]

    # Para aplicar kmeans
    X = anom.stack(new=['lon', 'lat']).hgt

    # Scores
    SSEScore(X, title='SSE',
             name_fig='SSE_score_' + str(a) + '_mes' + str(m),
             save=save)

    SilhScore(X, title='Silhouette score',
              name_fig='Silh_score_' + str(a) + '_mes' + str(m),
              save=save)

    pred, clusters = KM(X, nc)
    clusters[3, :].reshape(21, 17).T

    extent = [270, 320, -60, -20]
    levels = np.arange(-200, 250, 50)
    levels = [-150, -100, -75, -50, -15, 0, 15, 50, 75, 100, 150]
    cmap = cbar_t
    fig_size = [11, 5]
    xticks = np.arange(270, 330, 10)
    yticks = np.arange(-60, 0, 20)

    fig, axs = plt.subplots(nrows=2, ncols=4,
                            subplot_kw={'projection': ccrs.PlateCarree(
                                central_longitude=180)}, figsize=fig_size,
                            dpi=dpi)

    crs_latlon = ccrs.PlateCarree()
    for c in range(0, clusters.shape[0]):
        if c >= 4:
            c2 = 1
            c3 = c - 4
        else:
            c2 = 0
            c3 = c

        aux = clusters[c, :].reshape(21, 17).T

        axs[c2][c3].set_extent(extent, crs=crs_latlon)
        im = axs[c2][c3].contourf(mmonth.lon, mmonth.lat, aux,
                                  levels=levels,
                                  transform=crs_latlon,
                                  cmap=cmap, extend='both')

        axs[c2][c3].contour(mmonth.lon, mmonth.lat, aux,
                            linewidths=.8, levels=levels,
                            transform=crs_latlon,
                            colors='black')

        color_map = '#4B4B4B'
        axs[c2][c3].add_feature(cartopy.feature.LAND, facecolor='lightgrey',
                                edgecolor=color_map)
        axs[c2][c3].add_feature(cartopy.feature.COASTLINE, linewidth=0.5)
        axs[c2][c3].coastlines(color=color_map, linestyle='-', alpha=1)
        axs[c2][c3].gridlines(linewidth=0.3, linestyle='-')
        axs[c2][c3].set_xticks(xticks, crs=crs_latlon)
        axs[c2][c3].set_yticks(yticks, crs=crs_latlon)
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        axs[c2][c3].xaxis.set_major_formatter(lon_formatter)
        axs[c2][c3].yaxis.set_major_formatter(lat_formatter)
        axs[c2][c3].tick_params(labelsize=8)
        title = 'Cluster ' + str(c + 1)
        axs[c2][c3].set_title(title, fontsize=15)
        plt.tight_layout()

    fig.subplots_adjust(right=0.925)
    pos = fig.add_axes([0.935, 0.2, 0.012, 0.6])
    cbar = fig.colorbar(im, cax=pos, pad=0.1)
    if save:
        plt.savefig(out_dir + 'cluster_' + str(a) + '_' + str(m) + '.jpg', dpi=300)
        plt.close()
    else:
        plt.show()

    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.scatter(np.arange(1,len(pred)+1),pred+1)
    plt.title('Evolución clusters')
    ax.set_xlabel('dias')
    ax.set_ylabel('Cluster')
    if save:
        plt.savefig(
            out_dir + 'evocluster_' + str(a) + '_' + str(m) + '.jpg', dpi=300)
        plt.close()
    else:
        plt.show()

