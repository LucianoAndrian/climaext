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


my.wt = analyze.wavelet(temp[c('Age', 'Temperature')], my.series = 1, dt=1000)

# definir paso/s temporal --> python
