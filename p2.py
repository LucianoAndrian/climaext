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

def plotperiods(anio, titulo, name_fig, dpi, save,
                anioper_mc, anioper_ec, anioper_mc_tm, anioper_ec_tm,
                anioper_mf, anioper_ef):
    fig = plt.figure(figsize=(10, 5), dpi=dpi)
    ax = fig.add_subplot(111)

    plt.fill_between(x=np.linspace(1, 365, 365),
                     y1=perc_t['tx_99'].values, y2=perc_t['tm_01'].values,
                     color='#B5B5B5', alpha=1, label='p01-p99', linewidth=0.0)

    plt.fill_between(x=np.linspace(1, 365, 365),
                     y1=perc_t['tx_9'].values, y2=perc_t['tm_1'].values,
                     color='#F9EEEE', alpha=1, label='p10-p90', linewidth=0.0)
    # 2020
    ax.plot(anio['dj'], anio['tx'].values, label='2020', color='k', linewidth=1)
    ax.plot(anio['dj'], anio['tm'].values, label='2020', color='blue',
            linewidth=1)

    try:
        ax.scatter(anioper_mc['dj'], anioper_mc['tx'].values,
                   label='Muy Calido',
                   color='magenta', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_ec['dj'], anioper_ec['tx'].values,
                   label='Ext. Calido',
                   color='red', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_mc_tm['dj'], anioper_mc_tm['tm'].values,
                   label='Muy Calido Tn',
                   color='coral', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_ec_tm['dj'], anioper_ec_tm['tm'].values,
                   label='Ext. Calido Tn',
                   color='orange', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_mf['dj'], anioper_mf['tm'].values, label='Muy Frio',
                   color='k', linewidth=3)
    except:
        pass

    try:
        ax.scatter(anioper_ef['dj'], anioper_ef['tm'].values, label='Ext. Frio',
                   color='purple', linewidth=3)
    except:
        pass

    #setear ejes y titulo
    print(titulo)

    plt.legend()
    if save:
        plt.savefig(out_dir + name_fig +'.jpg')
        print('Save')
        plt.close('all')
    else:
        plt.show()

def plotItem3c(anio, titulo, dpi, save, per_pp, anioacum):
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

    print(titulo)
    #setar indices
    plt.legend()
    if save:
        plt.savefig(out_dir + name_fig +'.jpg')
        print('Save')
        plt.close('all')
    else:
        plt.show()
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
jn['tm']=jn['tm'].replace([213.3], 13.3)
# buscando otra vez
tm_tx = jn.loc[jn['tx'] < jn['tm']]
#nada
#ok

# outliers --------------------------------------------------------------------#
# Temperatura
# para cada dia:
# se calcula media y desvio
# se evalua que eventos superan el media+-3sd \ esto da un monton!
# para no revsisar uno por uno:
# se revisan si esos valores superan los registros historicos del smn
tm_out_count=0
tx_out_count=0
tm_min_smn = [6, 5, 1, -3.4, -7.5, -9.2, -8, -7, -5.4, -3.4, 1.2, 2.2]
tm_max_smn = [26.4, 24, 23.9, 22.2, 22.1, 18.5, 18.1, 19.5, 20.8, 21.6,
              22.5, 24.9]
tx_min_smn = [15.1, 15.6, 15, 11.7, 7.7, 6.2, 3.5, 6.3, 7.8, 11, 13.3, 16]
tx_max_smn = [42.1, 39.7, 37.6, 34.5, 31.8, 27.3, 30.8, 35.3, 37.0, 37.4,
              39.7, 41.8]


for mes in range(1, 13):
    for dia in range(1, 32):
        if (mes == 2) and (dia == 29):
            pass
        else:
            aux = jn.loc[(jn['dia'] == dia) & (jn['mes'] == mes)]
            media_tx = np.mean(aux.tx)
            sd3_tx = 3*np.std(aux.tx)
            tx_out = aux.loc[(aux.tx>media_tx+sd3_tx) |
                             (aux.tx<media_tx-sd3_tx)]

            if (len(tx_out)>0):
                if tx_out_count==0:
                    tx_out_df = checkEx(tx_out, media_tx, mes,
                                        tx_max_smn, tx_min_smn, 'tx')
                    if len(tx_out_df)>0:
                        tx_out_count=1
                else:
                    tx_out = checkEx(tx_out, media_tx, mes,
                                        tx_max_smn, tx_min_smn, 'tx')
                    if len(tx_out)>0:
                        tx_out_df = pd.concat([tx_out_df, tx_out], axis=0)


            media_tm = np.mean(aux.tm)
            sd3_tm = 3*np.std(aux.tm)
            tm_out = aux.loc[(aux.tm > media_tm + sd3_tm) |
                             (aux.tm < media_tm - sd3_tm)]

            if (len(tm_out)>0):
                if tm_out_count==0:
                    tm_out_df = checkEx(tm_out, media_tm, mes,
                                        tm_max_smn, tm_min_smn, 'tm')
                    if len(tm_out_df)>0:
                        tm_out_count=1
                else:
                    tm_out = checkEx(tm_out, media_tm, mes,
                                        tm_max_smn, tm_min_smn, 'tm')
                    if len(tm_out)>0:
                        tm_out_df = pd.concat([tm_out_df, tm_out], axis=0)

# >>> tm_out_df
# []
# >>> tx_out_df
#         estacion  dia  mes  anio    tx   tm    pp  ppc    h  dv    vx
# 199106     87548   15    4  1959  10.4  7.0  72.0  NaN  0.0  18  72.0
# un solo valor de tx se encuentra por debajo del registro historico del smn
# que llega hasta 1961, el valor no es tan extremo --> "queda"

#------------------------------------------------------------------------------#
# Precipitación
# controlado manualmente a partir de esto y lo acumulado diario con el smn
#ok
for a in range(1959,2021):
    for mes in range(1,13):
        ppa = jn.loc[(jn['anio'] == a) &
                           (jn['mes'] == mes)].sum().pp

        m = {'mes':[mes], 'anio':[a], 'ppa':[ppa]}
        if mes == 1:
            meses = pd.DataFrame(m)
        else:
            meses = pd.concat([meses, pd.DataFrame(m)], axis=0)

    if a == 1959:
        ppaf = meses
    else:
        ppaf = pd.concat([ppaf, meses], axis=0)

#------------------------------------------------------------------------------#
# 2 ---------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
# Calcular los percentiles diarios (1, 10, 90, 99) de tx y tm y los
# percentiles mensuales (95, 99) de la pp (no nula) para el período 1981-2010.
jn_81_10 = jn.loc[(jn['anio']>=1981)&(jn['anio']<=2010)]

# temp --------------------------#
for q in [.01, .10, .90, .99]:
    for mes in range(1,13):
        for dia in range(1,32):
            if (mes==2) and (dia==29):
                pass
            else:
                per = jn_81_10.loc[(jn_81_10['dia'] == dia) &
                                   (jn_81_10['mes'] == mes)].quantile(q)

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
        per = jn.loc[(jn['mes'] == mes) & (jn['pp']>0.1)].quantile(q)

        m = {'mes':[mes], 'pp_'+ str(q).split('.')[1]:[per.pp]}
        if mes == 1:
            meses = pd.DataFrame(m)
        else:
            meses = pd.concat([meses, pd.DataFrame(m)], axis=0)

    if q == .95:
        per_pp = meses
    else:
        per_pp = pd.concat([per_pp, meses.pp_99], axis=1)

#------------------------------------------------------------------------------#
# Plots punto 3
#a. Graficar la curva de temperatura observada en 2020-2021 y las curvas de los
# percentiles diarios de la climatología en un mismo gráfico.

a20 = jn.loc[jn['anio']==2020]
index = a20.loc[(a20['mes']==2) & (a20['dia']==29)].index
a20 = a20.drop(index[0])
a20['dj'] = range(1,366)
a21 = jn.loc[jn['anio']==2021]
a21['dj'] = range(1,366)

# tx
fig = plt.figure(figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)

plt.fill_between(x=np.linspace(1,365,365),
                 y1=perc_t['tx_99'].values, y2=perc_t['tx_01'].values,
                 color='#FF735B', alpha=1, label='p01-p99', linewidth=0.0)

plt.fill_between(x=np.linspace(1,365,365),
                 y1=perc_t['tx_9'].values, y2=perc_t['tx_1'].values,
                 color='#FFCFC2', alpha=1, label='p10-p90', linewidth=0.0)
# 2020
ax.plot(a20['tx'].values, label='2020', color='k', linewidth=3)
# 2021
ax.plot(a21['tx'].values, label='2021', color='#0500DE', linewidth=3)

#setear ejes y fechas en x
plt.legend()
if save:
    plt.savefig(out_dir + '_3a_tx.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()

# tm
fig = plt.figure(figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)

plt.fill_between(x=np.linspace(1,365,365),
                 y1=perc_t['tm_99'].values, y2=perc_t['tm_01'].values,
                 color='#1BBAC0', alpha=1, label='p01-p99', linewidth=0.0)

plt.fill_between(x=np.linspace(1,365,365),
                 y1=perc_t['tm_9'].values, y2=perc_t['tm_1'].values,
                 color='#BEDEDE', alpha=1, label='p10-p90', linewidth=0.0)
# 2020
ax.plot(a20['tm'].values, label='2020', color='k', linewidth=3)
# 2021
ax.plot(a21['tm'].values, label='2021', color='#FF0053', linewidth=3)

#setear ejes y fechas en x
plt.legend()
if save:
    plt.savefig(out_dir + '_3a_tm.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()

# 3b --------------------------------------------------------------------------#
# 2020
meses_mc, meses_ec, meses_mc_tm, meses_ec_tm, meses_mf, meses_ef\
    = PeriodosTemp(a20, perc_t)

a20per_mc = DetecPeriodsx3(meses_mc)
a20per_ec = DetecPeriodsx3(meses_ec)
a20per_mc_tm = DetecPeriodsx3(meses_mc_tm)
a20per_ec_tm = DetecPeriodsx3(meses_ec_tm)
a20per_mf = DetecPeriodsx3(meses_mf)
a20per_ef = DetecPeriodsx3(meses_ef)

plotperiods(a20, 'titulo', 'name_fig', 100, False,
                a20per_mc, a20per_ec, a20per_mc_tm, a20per_ec_tm,
                a20per_mf, a20per_ef)

# 2021
meses_mc, meses_ec, meses_mc_tm, meses_ec_tm, meses_mf, meses_ef\
    = PeriodosTemp(a21, perc_t)

a21per_mc = DetecPeriodsx3(meses_mc)
a21per_ec = DetecPeriodsx3(meses_ec)
a21per_mc_tm = DetecPeriodsx3(meses_mc_tm)
a21per_ec_tm = DetecPeriodsx3(meses_ec_tm)
a21per_mf = DetecPeriodsx3(meses_mf)
a21per_ef = DetecPeriodsx3(meses_ef)

plotperiods(a21, 'titulo', 'name_fig', 100, False,
                a21per_mc, a21per_ec, a21per_mc_tm, a21per_ec_tm,
                a21per_mf, a21per_ef)

# 3cd -------------------------------------------------------------------------#
def MonthMeanPP(data):
    for m in range(1, 13):
        aux = data.loc[(data['mes']==m) & (data['pp']>0.1)].mean().pp
        m2 = {'mes':[m], 'ppa':[aux]}
        if m == 1:
            meses = pd.DataFrame(m2)
        else:
            meses = pd.concat([meses, pd.DataFrame(m2)], axis=0)

    return meses

a20_pp_mm = MonthMeanPP(a20)
a21_pp_mm = MonthMeanPP(a21)

plotItem3c(a20, 'titulo', dpi, False, per_pp, a20_pp_mm)
plotItem3c(a21, 'titulo', dpi, False, per_pp, a21_pp_mm)

# 3e --------------------------------------------------------------------------#
for i in range(1981,2011):
    aux = jn_81_10.loc[jn_81_10['anio']==i]
    d = pd.DataFrame(aux.pp.values)
    if i == 1981:
        da = d
    else:
        da = pd.concat([da,d], axis=1)

fig = plt.figure(figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)

ax.plot(da.fillna(0).transform('cumsum').mean(axis=1), label='1981-2010',
         linewidth=2, color='k')
ax.plot(a20.dj.values, a20.fillna(0).transform('cumsum').pp.values,
            label='2020', linewidth=2)
ax.plot(a20.dj.values,a21.fillna(0).transform('cumsum').pp.values,
           label='2021', linewidth=2)

#setear ejes y fechas en x
plt.legend()
if save:
    plt.savefig(out_dir + '_3a_tx.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()
# 4 y 5 -----------------------------------------------------------------------#
for i in range(1960,2022):
    aux = jn.loc[jn['anio'] == i]
    tx_c_m = 0
    tm_c_m = 0
    pp_c_m = 0

    for m in range(1,13):
        tx_count = 0
        tm_count = 0
        pp_count = 0
        for d in range(1, 32):
            try:
                aux_d = aux.loc[(aux['mes'] == m) & (aux['dia'] == d)]

                aux_d_per = perc_t.loc[
                    (perc_t['mes'] == m) & (perc_t['dia'] == d)]

                aux2_d_per = per_pp.loc[
                    (per_pp['mes'] == m)]

                txaux = aux_d.tx.values[0] >= aux_d_per.tx_9[0]
                tmaux = aux_d.tm.values[0] <= aux_d_per.tm_1[0]
                ppaux = aux_d.pp.values[0] >= aux2_d_per.pp_95[0]

                if txaux:
                    tx_count += 1

                if tmaux:
                    tm_count += 1

                if ppaux:
                    pp_count += 1
            except:
                pass

        tx_c_m += tx_count
        tm_c_m += tm_count
        pp_c_m += pp_count

    aux2 = pd.DataFrame({'anio':[i],
                         'txper':[np.round(100*tx_c_m/365, 2)],
                         'tmper':[np.round(100*tm_c_m/365, 2)],
                         'ppper':[np.round(100*pp_c_m/365, 2)]})

    if i == 1960:
        anio_per = aux2
    else:
        anio_per = pd.concat([anio_per, aux2], axis=0)



fig = plt.figure(figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
txl = ax2.plot(np.arange(0, 62), anio_per.txper.values, color='firebrick',
               label='% T > p90', linewidth=2)
tml= ax2.plot(np.arange(0, 62), anio_per.tmper.values, color='limegreen',
              label='% T < p10', linewidth=2)
ppb = ax.bar(np.arange(0, 62), anio_per.ppper.values, color='dodgerblue',
             label='% PP > p95', linewidth=2)

ax2.set_ylim(0, 25)
ax.set_ylim(0, 5)
ax2.scatter(35, anio_per.loc[anio_per['anio']==1995].txper.values, color='red')
ax2.scatter(35,anio_per.loc[anio_per['anio']==1995].tmper.values, color='green')
ax2.annotate('1995',(33.5, anio_per.loc[anio_per['anio']==1995].tmper.values+1))

ax.bar(35, anio_per.loc[anio_per['anio']==1995].ppper.values, color='blue')
import matplotlib.patches as mpatches
ax2.legend(loc='upper right')
ppb = mpatches.Patch(color='dodgerblue', label='% PP > p95')
ax.legend(handles=[ppb], loc='upper right', bbox_to_anchor=(.83,1))
plt.show()
#
# 6 y 7 -----------------------------------------------------------------------#
for i in range(1960,2022):
    aux = jn.loc[jn['anio'] == i]
    tx_c_m = 0
    tm_c_m = 0
    pp_c_m = 0
    tx2_c_m = 0
    tm2_c_m = 0
    for m in range(1,13):
        tx_count = 0
        tm_count = 0
        pp_count = 0
        tx2_count = 0
        tm2_count = 0
        for d in range(1, 32):
            try:
                aux_d = aux.loc[(aux['mes'] == m) & (aux['dia'] == d)]

                txaux = aux_d.tx.values[0] >= 32
                tmaux = aux_d.tm.values[0] >= 20
                ppaux = aux_d.pp.values[0] >= 10

                txaux2 = aux_d.tx.values[0] <= 10
                tmaux2 = aux_d.tm.values[0] <= 0

                if txaux:
                    tx_count += 1

                if tmaux:
                    tm_count += 1

                if ppaux:
                    pp_count += 1

                if txaux2:
                    tx2_count += 1

                if tmaux2:
                    tm2_count += 1

            except:
                pass

        tx_c_m += tx_count
        tm_c_m += tm_count
        pp_c_m += pp_count
        tx2_c_m += tx2_count
        tm2_c_m += tm2_count

    aux2 = pd.DataFrame({'anio':[i],
                         'txper':[np.round(100*tx_c_m/365, 2)],
                         'tmper':[np.round(100*tm_c_m/365, 2)],
                         'ppper':[np.round(100*pp_c_m/365, 2)],
                         'txper2': [np.round(100 * tx2_c_m / 365, 2)],
                         'tmper2': [np.round(100 * tm2_c_m / 365, 2)]})

    if i == 1960:
        anio_per = aux2
    else:
        anio_per = pd.concat([anio_per, aux2], axis=0)


fig = plt.figure(figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
txl = ax2.plot(np.arange(0, 62), anio_per.txper.values, color='firebrick',
               label='% Tx > 32ºC', linewidth=2)
tml= ax2.plot(np.arange(0, 62), anio_per.tmper.values, color='limegreen',
              label='% Tm > 20ºC', linewidth=2)
ppb = ax.bar(np.arange(0, 62), anio_per.ppper.values, color='dodgerblue',
             label='% PP > 10 mm', linewidth=2)

ax2.set_ylim(-10, 20)
ax.set_ylim(0, 30)
import matplotlib.patches as mpatches
ax2.legend(loc='upper right')
ppb = mpatches.Patch(color='dodgerblue', label='% PP > 10 mm')
ax.legend(handles=[ppb], loc='upper right', bbox_to_anchor=(.83,1))
plt.show()


fig = plt.figure(figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
txl = ax.plot(np.arange(0, 62), anio_per.txper2.values, color='firebrick',
               label='% Tx < 10ºC', linewidth=2)
tml= ax.plot(np.arange(0, 62), anio_per.tmper2.values, color='limegreen',
              label='% Tm < 0ºC', linewidth=2)

ax.set_ylim(0, 20)
ax.legend(loc='upper right')
plt.show()

# 8 ---------------------------------------------------------------------------#
# se calcularon
# FD (frost days) tm<0
# TR (tropical nights) tm>20
# TN10p TX90p
# R10mm

# Parte 2 ---------------------------------------------------------------------#
# R no anda, no quiere, no puede, quien sabe.
from scipy import stats
from scipy.stats import genextreme as gev
from pyextremes import EVA, get_extremes

jn2 = jn[['dia','mes', 'anio', 'tx', 'tm']]
jn2 = jn2.loc[jn2['anio']>1959]
jn2['date'] = pd.to_datetime(
    dict(year=jn2.anio, month=jn2.mes, day=jn2.dia))

# a.
jn2tx = jn2.loc[jn2['mes']==1]
jn2tm = jn2.loc[jn2['mes']==7]

txserie = jn2tx['tx'].squeeze()
txserie = txserie.set_axis(pd.to_datetime(jn2tx['date'].values))

tmserie = jn2tm['tm'].squeeze()
tmserie = tmserie.set_axis(pd.to_datetime(jn2tm['date'].values))

txx = get_extremes( ts=txserie, method="BM", extremes_type="high")
plt.plot(jn2.date, jn2.tx)
plt.scatter(txx.index, txx, c='red')
plt.show()

tmn = get_extremes( ts=tmserie, method="BM", extremes_type="low",
                    errors='ignore')
plt.plot(jn2.date, jn2.tm)
plt.scatter(tmn.index, tmn, c='red')
plt.show()

# b c -------------------------------------------------------------------------#
#tx
modeltx = EVA(txserie.loc[~np.isnan(txserie)])
modeltx.get_extremes(method="BM", block_size="365.2425D", extremes_type="high")
modeltx.fit_model()
# periodos de retorno
summarytx = modeltx.get_summary( return_period=[2, 10, 50, 100], alpha=0.95,
                             n_samples=1000) # estima la sig. con bootstrap
print(summarytx)
modeltx.plot_diagnostic(alpha=0.95)
plt.show()

# tn
modeltm = EVA(tmserie.loc[~np.isnan(tmserie)])
modeltm.get_extremes(method="BM", block_size="365.2425D", extremes_type="low",
                     errors='ignore')
modeltm.fit_model()
# periodos de retorno
summarytm = modeltm.get_summary( return_period=[2, 10, 50, 100], alpha=0.95,
                             n_samples=1000) # estima la sig. con bootstrap
print(summarytm)
modeltm.plot_diagnostic(alpha=0.95)
plt.show()
# 2 KS con bootstrap ----------------------------------------------------------#
# TXx
try: # Puede pasar que la distribucion no se ajuste a genextreme y no tiene c
    c = list((modeltx.distribution.mle_parameters.values()))[0]
    loc = list((modeltx.distribution.mle_parameters.values()))[1]
    scale = list((modeltx.distribution.mle_parameters.values()))[2]
except:
    print('FALTA UN PARAMETRO!')

kstx = stats.kstest(modeltx.extremes.values, 'genextreme',
                    args=(c, loc, scale)).statistic
# Bootstrap
gevds = gev(c, loc, scale)
ks_bt_tx = []
for n in range(5000):
    # random a partir de la original
    aux = gevds.rvs(size=len(model.extremes.values))
    gf2 = gev.fit(aux)  # gev
    # ks test
    ks_test = stats.kstest(aux, 'genextreme', args=(gf2[0], gf2[1], gf2[2]))
    ks_bt_tx.append(ks_test.statistic)

pbt_tx = 100*sum(ks_bt_tx > kstx)/5000

plt.hist(ks_bt_tx, bins=10,rwidth=1)
plt.show()

# TNn
try:
    c = list((modeltm.distribution.mle_parameters.values()))[0]
    loc = list((modeltm.distribution.mle_parameters.values()))[1]
    scale = list((modeltm.distribution.mle_parameters.values()))[2]
except:
    print('FALTA UN PARAMETRO!')

kstm = stats.kstest(modeltm.extremes.values, 'genextreme',
                    args=(c, loc, scale)).statistic
# Bootstrap
gevds = gev(c, loc, scale)
ks_bt_tm = []
for n in range(5000):
    # random a partir de la original
    aux = gevds.rvs(size=len(model.extremes.values))
    gf2 = gev.fit(aux)  # gev
    # ks test
    ks_test = stats.kstest(aux, 'genextreme', args=(gf2[0], gf2[1], gf2[2]))
    ks_bt_tm.append(ks_test.statistic)

pbt_tm = 100*sum(ks_bt_tm > kstm)/5000

plt.hist(ks_bt_tm, bins=50,rwidth=1)
plt.show()

################################################################################
################################################################################



