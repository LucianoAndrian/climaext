# P0
#------------------------------------------------------------------------------#
library(WaveletComp)
source('funcionesmetodos.R')
#------------------------------------------------------------------------------#
data_dir = '/home/luciano.andrian/doc/climaext/p0/data/'
out_dir = '/home/luciano.andrian/doc/climaext/p0/'
save = F
#----------------------------------------------------------------------------- #
# 2 Wavelet
temp = read.table(paste(data_dir, 'edc3deuttemp2007_Resunamierda.txt', sep=''), 
                  header = T, fill=T)


df = data.frame(c(temp$Temperature))

# la temperatura tiene 3 valores faltantesm de mas de 5mil, reemplazo por la media
temp$Temperature[which(is.na(temp$Temperature))]=mean(temp$Temperature, na.rm = T)

my.wt = analyze.wavelet(temp, my.series = 5, dt=1, dj = 1/10,
                        lowerPeriod = 10, upperPeriod = 4000, n.sim = 10)

pdf("temp_wavelet.pdf", width = 7, height = 5)
wt.image(my.wt, 
         color.key="interval", 
         legend.params=list(lab="Espectro de poder de wavelet"),
         siglvl=0.05, # SIGNIFICANCIA
         col.contour = 'black',
         plot.ridge = FALSE, #AGREGA CONTORNO EXTRA DE LOS PUNTOS DE MAX VARIANZA
         color.palette = 'rainbow(n.levels,start=1,end=.65)' ,
         main= 'Temp. 800yk',
         timelab = 'Tiempo',
         periodlab= 'Periodo')
dev.off()
 
co2 = read.table(paste(data_dir, 'edc3-composite-co2-2008-noaa.txt', sep=''), 
                  header = T, fill=T, skip=275)


my.wt = analyze.wavelet(co2, my.series = 2, dt=1, dj = 1/10,
                        lowerPeriod = 10)

pdf("co2_wavelet.pdf", width = 7, height = 5)
wt.image(my.wt, 
         color.key="interval", 
         legend.params=list(lab="Espectro de poder de wavelet"),
         siglvl=0.05, # SIGNIFICANCIA
         col.contour = 'black',
         plot.ridge = FALSE, #AGREGA CONTORNO EXTRA DE LOS PUNTOS DE MAX VARIANZA
         main= 'CO2 800yk',
         timelab = 'Tiempo',
         periodlab= 'Periodo')
dev.off()




#- prueba Wavelet con pasos temporales distintos -#
# carga de datos
datos <- read.table(paste(data_dir, "Tmensual_Aeroparque_ejemplo.txt", sep = ''), sep="",header=FALSE)
x1 = datos[,1][0:200]
x2 = datos[,1][200:428][c(T,F)]
x = as.data.frame(append(x1, x2))
plot(x[,1], type='l')
#con la función analyze.wavelet obtengo el wavelet de mis datos
my.wt = analyze.wavelet(x, my.series=1,dt=1, lowerPeriod = 1, upperPeriod = 32)

wt.image(my.wt, 
         color.key="interval", 
         legend.params=list(lab="Espectro de poder de wavelet"),
         siglvl=0.05, # SIGNIFICANCIA
         col.contour = 'black',
         plot.ridge = FALSE, #AGREGA CONTORNO EXTRA DE LOS PUNTOS DE MAX VARIANZA
         color.palette = 'rainbow(n.levels,start=1,end=.65)' ,
         main= 'Temp 800yk',
         timelab = 'Tiempo [años]',
         periodlab= 'Periodo [años]')





