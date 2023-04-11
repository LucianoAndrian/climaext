# P0
#----------------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#----------------------------------------------------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p0/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p0/'
save = True
dpi = 300
#----------------------------------------------------------------------------------------------------------------------#
# 2
temp = pd.read_csv(data_dir + 'edc3deuttemp2007.txt', sep='\s+', skiprows=91) # \s+ separador de más de 1 espacio
temp_age = temp.loc[:,'Age']
temp_data = temp.loc[:,'Temperature']
co2 =  pd.read_csv(data_dir + 'edc3-composite-co2-2008-noaa.txt', sep='\s+', skiprows=275)
co2_age = co2.loc[:,'gas_ageBP']
co2_data = co2.loc[:,'CO2']

# Plot
fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
co2_tl = ax.plot(co2_age, co2_data, '-o', label='CO2 [ppm]', color='turquoise', linewidth=2,
                 markersize=.5, markerfacecolor='k', markeredgecolor='k')
temp_tl = ax2.plot(temp_age, temp_data, '-o', label='Temp. [ºC]', color='orange', linewidth=2,
                   markersize=.5, markerfacecolor='k', markeredgecolor='k')

ax.set_xlim(temp_age[12],temp_age.iloc[-1])
xticks = np.arange(0,temp_age.iloc[-1],1e5)
xlab=[]
for c,x in enumerate(xticks):
    xstr=format(x, '.0e').split('e')
    xlab.append(xstr[0] + '$x10^{5}$')
ax.set_xticks(xticks, xlab)
ax.set_xlabel('Años (bp)')

ax.set_ylim(100, 300)
ax.set_ylabel('CO2 [ppm]')
ax2.set_ylim(-12,8)
ax2.set_ylabel('Temp. [ºC]')

lns = co2_tl + temp_tl
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper right')

if save:
    plt.savefig(out_dir + 'p0_c.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()


# time step check
def CheckTimeStep(serie):
    diff = []
    for i in range(0, len(serie)):
        t0 = np.abs(serie[i])
        try:
            t1 = np.abs(serie[i + 1])
        except:
            return diff
        if t0 > t1:
            t_dif = t0 - t1
        else:
            t_dif = t1 - t0

        diff.append(t_dif)

plt.plot(CheckTimeStep(co2_age));plt.show()
plt.plot(CheckTimeStep(temp_age));plt.show()

# Plot
fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
ax2 = ax.twinx()
co2_tl = ax.plot(co2_age[0:-1], CheckTimeStep(co2_age), '-o', label='CO2 [ppm]', color='turquoise', linewidth=2,
                 markersize=.5, markerfacecolor='k', markeredgecolor='k')
temp_tl = ax2.plot(temp_age[0:-1], CheckTimeStep(temp_age), '-o', label='Temp. [ºC]', color='orange', linewidth=2,
                   markersize=.5, markerfacecolor='k', markeredgecolor='k')

ax.set_ylim(0, 6000)
ax.set_ylabel('CO2 pasos temporales')
ax2.set_ylim(0,1200)
ax2.set_ylabel('Temp. pasos temporales')

lns = co2_tl + temp_tl
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left')

if save:
    plt.savefig(out_dir + 'p0_g_aux.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()


# Plot
fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
#ax2 = ax.twinx()
co2_tl = ax.plot(CheckTimeStep(co2_age), '-o', label='CO2 [ppm]', color='turquoise', linewidth=2,
                 markersize=.5, markerfacecolor='k', markeredgecolor='k')
# temp_tl = ax2.plot(CheckTimeStep(temp_age), '-o', label='Temp. [ºC]', color='orange', linewidth=2,
#                    markersize=.5, markerfacecolor='k', markeredgecolor='k')

ax.set_ylim(0, 6000)
ax.set_ylabel('CO2 pasos temporales')
# ax2.set_ylim(0,1200)
# ax2.set_ylabel('Temp. pasos temporales')

lns = co2_tl + temp_tl
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left')

if save:
    plt.savefig(out_dir + 'p0_g_aux_temp.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()


# Plot
fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
#ax2 = ax.twinx()
# co2_tl = ax.plot(CheckTimeStep(co2_age), '-o', label='CO2 [ppm]', color='turquoise', linewidth=2,
#                  markersize=.5, markerfacecolor='k', markeredgecolor='k')
temp_tl = ax.plot(CheckTimeStep(temp_age), '-o', label='Temp. [ºC]', color='orange', linewidth=2,
                   markersize=.5, markerfacecolor='k', markeredgecolor='k')

ax.set_ylim(0,1200)
ax.set_ylabel('Temp. pasos temporales')

lns = co2_tl + temp_tl
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc='upper left')

if save:
    plt.savefig(out_dir + 'p0_g_aux_co2.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()
#----------------------------------------------------------------------------------------------------------------------#
# 3
save=True
dpi=300
# Darrigo
#el archivo es una mierda:
temp = pd.read_csv(data_dir + 'auxdarrigo2006.txt', sep='\s+', skipfooter=150, engine='python') # \s+ separador de más de 1 espacio
temp2 = pd.read_csv(data_dir + 'auxdarrigo2006.txt', sep='\s+',skiprows= 1146, skipfooter=9, engine='python') # \s+ separador de más de 1 espacio
# renombrar las columas de temp xq quedaron mal.
temp.rename(columns={'RCSrecon':'borrar'}, inplace=True)
temp.rename(columns={'STDrecon':'RCSrecon'}, inplace=True)
temp.rename(columns={'NHLandObs':'STDrecon'}, inplace=True)
temp = temp.drop('borrar', axis=1)

temp2 = temp2.drop('NHLandObs', axis=1)

temp_darrigo = temp.append(temp2)

# Mann-Jones
temp_mj = pd.read_csv(data_dir + 'jonesmannrogfig4a.txt', sep='\s+',skiprows=8, skipfooter=2002, engine='python') # \s+ separador de más de 1 espacio
temp_mj = temp_mj.replace(-99.99, np.nan)

temp_mj_sm= pd.read_csv(data_dir + 'jonesmannrogfig4a.txt', sep='\s+',skiprows=2011, engine='python') # \s+ separador de más de 1 espacio
temp_mj_sm = temp_mj_sm.replace(-99.99, np.nan)
#Moberg
temp_moberg = pd.read_csv(data_dir + 'nhtemp-moberg2005.txt', sep='\s+',skiprows=92) # \s+ separador de más de 1 espacio

# CRU descargado de MetOffice xq lo otro está hecho para torturar gente
temp_cru = pd.read_csv(data_dir + 'HadCRUT.5.0.1.0.analysis.summary_series.northern_hemisphere.annual.csv', sep=',') # \s+ separador de más de 1 espacio


#Plot de todas las series juntas

# Plot
fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
#moberg
ax.plot(temp_moberg.loc[:,temp_moberg.columns[0]], temp_moberg.loc[:,temp_moberg.columns[1]], label='Moberg et al. 2005', linewidth=2, color='firebrick')
#mj
ax.plot(temp_mj.loc[:,temp_mj.columns[0]], temp_mj.loc[:,temp_mj.columns[1]], linewidth=1, alpha=0.2, color='dodgerblue')
ax.plot(temp_mj_sm.loc[:,temp_mj_sm.columns[0]], temp_mj_sm.loc[:,temp_mj_sm.columns[1]], label='Mann & Jones 2003', linewidth=2,color='dodgerblue')
#darrigo
ax.plot(temp_darrigo.loc[:,temp_darrigo.columns[0]], temp_darrigo.loc[:,temp_darrigo.columns[1]], label="STD D'arrigo et al. 2006", linewidth=2, color='green')
ax.plot(temp_darrigo.loc[:,temp_darrigo.columns[0]], temp_darrigo.loc[:,temp_darrigo.columns[2]], label="RCS D'arrigo et al. 2006", linewidth=2, color='lime')
#cru
ax.plot(temp_cru.loc[:,temp_cru.columns[0]], temp_cru.loc[:,temp_cru.columns[2]], label="HadCRUT 5.0.1.0", linewidth=2, color='magenta')
ax.axhline(y=0, color='k')
ax.grid()
ax.legend(loc='upper left')
ax.set_ylim(-2,2)
ax.set_ylabel("T' [ºC]")
ax.set_ylabel('Años')
ax.axvspan(900, 1300, facecolor='red', alpha=0.3)
ax.axvspan(1300, 1850, facecolor='blue', alpha=0.3)
if save:
    plt.savefig(out_dir + 'p0_ej3.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()

#Calentamiento medieval 900-1300
#Pequeña edad de hielo (q nombre pedorro) 1300-1850
fig = plt.figure(1, figsize=(10, 5), dpi=dpi)
ax = fig.add_subplot(111)
#moberg
ax.plot(temp_moberg.loc[:,temp_moberg.columns[0]], temp_moberg.loc[:,temp_moberg.columns[1]], label='Moberg et al. 2005', linewidth=2, color='firebrick')
#mj
ax.plot(temp_mj.loc[:,temp_mj.columns[0]], temp_mj.loc[:,temp_mj.columns[1]], linewidth=1, alpha=0.2, color='dodgerblue')
ax.plot(temp_mj_sm.loc[:,temp_mj_sm.columns[0]], temp_mj_sm.loc[:,temp_mj_sm.columns[1]], label='Mann & Jones 2003', linewidth=2,color='dodgerblue')
#darrigo
ax.plot(temp_darrigo.loc[:,temp_darrigo.columns[0]], temp_darrigo.loc[:,temp_darrigo.columns[1]], label="STD D'arrigo et al. 2006", linewidth=2, color='green')
ax.plot(temp_darrigo.loc[:,temp_darrigo.columns[0]], temp_darrigo.loc[:,temp_darrigo.columns[2]], label="RCS D'arrigo et al. 2006", linewidth=2, color='lime')
#cru
ax.plot(temp_cru.loc[:,temp_cru.columns[0]], temp_cru.loc[:,temp_cru.columns[2]], label="HadCRUT 5.0.1.0", linewidth=2, color='magenta')
ax.axhline(y=0, color='k')
ax.grid()
ax.legend(loc='upper left')
ax.set_ylim(-1.5,1.5)
ax.set_xlim(850,2023)
ax.set_ylabel("T' [ºC]")
ax.set_ylabel('Años')
ax.axvspan(900, 1300, facecolor='red', alpha=0.3)
ax.axvspan(1300, 1850, facecolor='blue', alpha=0.3)
if save:
    plt.savefig(out_dir + 'p0_ej3_zoom.jpg')
    print('Save')
    plt.close('all')
else:
    plt.show()

#Para seguir perdiendo tiempo:
# Test de media entre el calentamiento medieval y el del siglo xx...

def selecteras(df, low, top):
    return df.loc[(df[df.columns[0]]>=low)&(df[df.columns[0]]<top)]

temp_darrigo_cm = selecteras(temp_darrigo,1300,1851)
temp_moberg_cm = selecteras(temp_moberg,1300,1851)
temp_mj_sm_cm = selecteras(temp_mj_sm,1300,1851)

temp_darrigo_xx = selecteras(temp_darrigo,1900,2001)
temp_moberg_xx = selecteras(temp_moberg,1900,2001)
temp_mj_sm_xx = selecteras(temp_mj_sm,1900,2001)
temp_cru_xx = selecteras(temp_cru,1900,2001)


def meantest(data1, data2):
    import scipy.stats
    # t
    x1 = np.mean(data1)
    s1 = np.std(data1)
    n1 = len(data1)

    x2 = np.mean(data2)
    s2 = np.std(data2)
    n2 = len(data2)
    t = (x1-x2)/np.sqrt((s1**2/n1)+(s2**2/n2))

    #df
    num = ((s1**2/n1)+(s2**2/n2))**2
    den = (((s1**2/n1)**2)/(n1-1)) + (((s2**2/n2)**2)/(n2-1))
    df = num/den

    t_teorico = scipy.stats.t.ppf(0.95, df)
    return t, t_teorico, x1, x2, s1, s2, n1, n2

t, t_teo, x1_darrigo1, x2_darrigo1, s1, s2, n1, n2 = meantest(
    temp_darrigo_cm.loc[:,temp_darrigo_cm.columns[1]],
    temp_darrigo_xx.loc[:,temp_darrigo_xx.columns[1]])
print('darrigo1: ')
print('t:' + str(t))
print('t_teo:' + str(t_teo))

t, t_teo, x1_darrigo2, x2_darrigo2, s1, s2, n1, n2 = meantest(
    temp_darrigo_cm.loc[:,temp_darrigo_cm.columns[2]],
    temp_darrigo_xx.loc[:,temp_darrigo_xx.columns[2]])
print('darrigo2: ')
print('t:' + str(t))
print('t_teo:' + str(t_teo))

t, t_teo, x1_moberg, x2_moberg, s1, s2, n1, n2 = meantest(
    temp_moberg_cm.loc[:,temp_moberg_cm.columns[1]],
    temp_moberg_xx.loc[:,temp_moberg_xx.columns[1]])
print('moberg: ')
print('t:' + str(t))
print('t_teo:' + str(t_teo))

t, t_teo, x1_mj, x2_mj, s1, s2, n1, n2 = meantest(
    temp_mj_sm_cm.loc[:,temp_mj_sm_cm.columns[1]],
    temp_mj_sm_xx.loc[:,temp_mj_sm_xx.columns[1]])
print('mj: ')
print('t:' + str(t))
print('t_teo:' + str(t_teo))

# media de todas las bases de datos en cm
aux = {'darrigo1':temp_darrigo_cm.loc[:,temp_darrigo_cm.columns[1]],
       'darrigo2':temp_darrigo_cm.loc[:,temp_darrigo_cm.columns[1]],
       'moberg':temp_moberg_cm.loc[:,temp_moberg_cm.columns[1]],
       'mj':temp_mj_sm_cm.loc[:,temp_mj_sm_cm.columns[1]]}
aux_dataframe = pd.DataFrame(aux)
#hay nans al comienzo y final de algunas series
all_data_mean_cm = aux_dataframe.mean(axis=1, skipna=True)

t, t_teo, x1, x2, s1, s2, n1, n2 = meantest(
    all_data_mean_cm,
    temp_cru_xx.loc[:,temp_cru_xx.columns[1]])

print('todo vs cru: ')
print('t:' + str(t))
print('t_teo:' + str(t_teo))
