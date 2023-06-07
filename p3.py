# practica 3
################################################################################
# sólo ploteos
################################################################################
import xarray as xr
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.feature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
import warnings
warnings.filterwarnings("ignore")
################################################################################
data_dir = '/home/luciano.andrian/doc/climaext/p3/ncfiles/'
out_dir = '/home/luciano.andrian/doc/climaext/p3/salidas/'

save = True
dpi = 300
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
         color_map, cmap):

    import matplotlib.pyplot as plt
    import cartopy.feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    import cartopy.crs as ccrs
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
    cb.ax.tick_params(labelsize=8)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor=color_map)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=0.5, zorder=17)
    # ax.add_feature(cartopy.feature.COASTLINE)
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

#for
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
            name_fig = 'pp_anom_' + FindMonth(f) + '_' + FindYear(f) + '_' + variable
            scale = [-100, -75, -50, -25, -5, 0, 5, 25, 50, 75, 100]
            cbar = cbar_pp

        if variable == 'pp':
            data = data.rename({'prate': 'precip'})
            data *= (365 / 12)

        Plot(data, data.precip[0, :, :], scale, save, dpi,
             title, name_fig, out_dir, 'gray', cbar)

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
             title, name_fig, out_dir, 'gray', cbar)
################################################################################

