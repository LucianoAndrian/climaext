###############################################################
#                         FUNCIONES                           #
###############################################################

######ANALISI TENDENCIA####
analisis_tendencia<-function(datos){
  
  readline("recordar que laa 1er columna debe contener los tiempos y la 2da los datos")
  
  serie<-data.frame(datos[which(!is.na(datos[,2])),1],datos[which(!is.na(datos[,2])),2]) 
  serie[,1]<-seq(1:length(serie[,1])) #IMPORTANTE XQ SINO AFECTA EL LM
  ajuste1<-lm(serie[,2]~serie[,1],data=serie) #Y , X!!
  tendencia<-ajuste1$coefficients[1]+ajuste1$coefficients[2]*serie[,1]  #contruccion de la recta y=ax+b... coef[2]x+coef[1]
  
  ff<-readline("forma 1 centrado en cero o 2 en valores de la variable--(1 o 2)?: ")
  if(ff==1){
    datosf<-serie[,2] - tendencia
    plot(ts(datosf))
  } else {
    datosf<-serie[,2]- (ajuste1$coefficients[2]*serie[,1])
    plot(ts(datosf))
  }
  return(datosf)
}



######PROMEDIO MOVIL######
#DESCRIPCION
##Metodos estadisticos 2018
##Programa para realizar promedios moviles con cantidad impar de pesos, pensado para centrado.
##La serie de datos debe ser una columna y debe estar completa, y los
###tiempos deben estar en la primera columna. Luego el software pedira
###cual es la columna a la cual deseas aplicarle Prom Movil.

#\m/Luciano Andrian\m/#
#funcion:
#para los promedios moviles pesados tiene cargados coeficientes para 3 5 7 11 y 13
#pregunta si se quiere usar estos o ingresar nuevos
#si se selecciona un promedio pesado que no es uno de esos directamente hay que cargar los coeficientes
#pregunta antes de guardar el grafico
#

promedios_moviles<-function(x){
  serie<-x
  ####Calculo media y desv. estand. de la serie original
  orig<-as.integer(readline(prompt='Ingrese el nro de la columna donde esta su serie en el archivo:  '))   ###('Ingrese el nro de la columna donde esta su serie en el archivo: ')
  medorig<-mean(serie[,orig])
  ###Para ver algunos valores de la serie
  may<-max(serie[,orig])
  miy<-min(serie[,orig])
  maxi<-max(serie[,1])
  mixi<-min(serie[,1])
  
  ####
  print(' ')
  print(paste('Valor medio de la serie original: ',round(medorig,2)))
  
  dsorig<-(sd(serie[,orig])*(length(serie[,orig])-1))/(length(serie[,orig])) #chequear para que el 1
  print(paste('Desv. Estand. de la serie original: ',round(dsorig,2)))
  print(' ')
  
  pasos<- as.integer(readline(prompt='Ingrese la cantidad de Analisis con promedios moviles a realizar: '))
  print(pasos)
  
  n <- dim(serie)[1]
  newserie <- matrix(NA, n, pasos)
  
  cont<-1
  seguir<-1
  
  while (cont<=pasos)
  {
    if (seguir==1){
      ####Datos de entrada
      cant<-as.integer(readline(prompt='Ingrese el tamano de la ventana centrada a promediar (obs: debe ser impar)'))
      print(cant)
      print('Que tipo de peso desea aplicar?:')
      print('1 - Promedio movil ordinario')
      print('2 - Ingresar los pesos manualmente')
      print(' ')
      aux<-as.integer(readline(prompt='Su eleccion ha sido: '))
      print(aux)
      
      if (aux!=1 & aux!=2){
        print('Eleccion incorrecta')
      }
      peso3<-c(0.25,0.5,0.25)
      peso5<-c(0.1,0.2,0.4,0.2,0.1)
      peso7<-c(0.05,0.1,0.2,0.3,0.2,0.1,0.05)
      peso11<-c(0.01,0.02,0.05,0.1,0.15,0.34,0.15,0.1,0.05,0.02,0.01)
      peso13<-c(0.005,0.01,0.015,0.03,0.1,0.15,0.38,0.15,0.1,0.03,0.015,0.01,0.005)
      peso = rep(NA, cant)
      
      if (aux==1){
        for (i in 1:cant){
          peso[i]<-1/cant
        }
      } else if (aux==2){
        if(cant==3|cant==5|cant==7|cant==11|cant==13){
          pesos<-readline("Puede usar coeficientes ya cargados, desea hacerlo?(si/no): ")
          if(pesos=="si" & cant==3){
            peso<-peso3
          } else if(pesos=="si" & cant==5){
            peso<-peso5
          } else if(pesos=="si" & cant==7){
            peso<-peso7
          } else if(pesos=="si" & cant==11){
            peso<-peso11
          } else if(pesos=="si" & cant==13){
            peso<-peso13
          } else if(pesos=="no"){
            for (i in 1:cant){
              peso[i]<-as.numeric(readline(prompt=paste('Ingrese el peso de la variable X', i, ':')))
            }
            
          } else { 
            for (i in 1:cant){
              peso[i]<-as.numeric(readline(prompt=paste('Ingrese el peso de la variable X', i, ':')))
            } 
          }
        } else { 
          for (i in 1:cant){
            peso[i]<-as.numeric(readline(prompt=paste('Ingrese el peso de la variable X', i, ':')))
          }  
        }
      }
      
      
      #defino k como el radio de la ventana
      k <- (cant-1)/2
      
      ###Aplico Promedios m???viles a mi serie
      if (aux==0 || aux==1 || aux==2){
        col<-as.integer(readline(prompt='Ingrese el nro de la columna donde est??? su serie en el archivo: '))
        dim=dim(serie)
        for (i in (k+1):(dim[1]-k)){
          sum=0
          for (j in -k:k){
            sum<-sum + serie[i+j, col]*peso[j+k+1]
          }
          
          newserie[i,cont]<-sum
        }
        
        print(paste('Su serie pesada se encuentra en la columna ',cont,' de la variable newserie'))
        
        ##Grafico la serie original y la pesada
        grafico<-readline("Desea guardar el grafico de la serie pesada? (si/no): ")
        if(grafico=="si"){
          nombre<-readline("Nombre de la imagen a guardar: ")
          png(filename=paste(cont,nombre,".png",sep=""), width=5500, height=4000, res=600)
          plot(serie[,1],serie[,col],lwd=2,type='l',main=paste('Serie original y Serie pesada con periodos mayores a ',cant),xlab='tiempo',ylab='variable')
          lines(serie[,1],newserie[,cont],col='red',lwd=2.5)
          legend('topleft',legend=c('serieoriginal','seriepesada'),col=c('black','red'),lty=c(1,1),lwd=c(2,2.5))
          dev.off()
        }
        
        
        ### Calculo media y desv. stand.
        media<-mean(newserie[,cont], na.rm=TRUE)
        print(paste('El valor medio de la serie es: ',media))
        ds<-sd(newserie[,cont],1)
        print(paste('El desv???o estandar de la serie es: ',ds))
        print(' ')
      }
      
      cont<-cont+1
      if ((cont-1)!=pasos){
        seguir<-as.integer(readline(promp='Desea continuar con el siguiente promedio movil? 0-No; 1-Si ?: '))
        print(' ')
      }
    }
    else{
      print('%%%%%%%%%%%%%%%%%%%')
      print('Analisis interrumpido')
      break
    }
    
    if ((cont-1)==pasos){
      print('%%%%%%%%%%%%%%%%%%%')
      print('Analisis finalizado') 
    }
  }
  
  
  newserie
  return<-newserie
}

####MARONA####
#
# Metodos estadisticos 2019 #
#      SCRIP DE CLASE       #
## MODIFICADO POR LUCIANO ANDRIAN ##
#
#### Test de Maronna-Yohai 
#### SCRIP DE CLASE CONVERTIRDO EN FUNCION, CALCULA EL ESTADISTICO, TESTEA Y GRAFICA
## AHORA PREGUNTA POR MAXIMOS SECUNDARIOS TAMBIEEEN, SOLO PARA DATOS ANUALES, NO PROGRAMADO QUE SI NO ES ANUAL NO PREGUNTE
#SI NO EXISTEN SEGUNDOS MAXIMOS TODO OK.
#FALTA HACERLO PARA DIARIOS, SOBRE TODO EL GRAFICADO Q ES CON GGPLOT
#FALTA HACERLO PARA DATOS MENSUALES.(PUEDE SER IGUAL QUE AÑOS...)
### RESULTADOS DEVUELTOS EN FORMA DE LISTA

# La serie de referencia es X
# La serie en la que se quiere identificar la ocurrencia de una cambio es Y


marona<-function(x){
  y<-readline("Formato de fecha, anual: A o diario D: " )
  VAR <- x
  library(ggplot2)
  
  #('Calcula dimensiones')
  dimension<-dim(VAR)
  N<-dimension[1]
  L<-dimension[2]
  
  PX <- VAR[,2]
  PY <-VAR[,3]
  
  #'Estandariza las series' 
  X <- (PX - mean(PX))/sd(PX)
  Y <- (PY - mean(PY))/sd(PY)  
  
  # Defino vectores que voy a crear 
  XX <- vector()
  YY <- vector()
  XAC <- vector()
  YAC <- vector()
  
  #'Genera la serie de valores acumulados'
  
  XX[1] <- X[1]
  YY[1] <- Y[1]
  XAC[1] <- X[1]
  YAC[1] <- Y[1]
  
  for (J in 2:N) {
    YAC[J] <- YAC[J-1]+Y[J];
    XAC[J] <- XAC[J-1]+X[J];
    XX[J] <- XAC[J]/J;
    YY[J] <- YAC[J]/J;  
  }
  
  #'Calcula los valores medios de ambas variables') 
  
  XXM <- XX[N]
  YYM <- YY[N]
  
  #'Calcula varianza y covarianza') 
  
  SX <- var(X)*(N-1)
  SY <- var(Y)*(N-1)
  
  SXYA <- cov(X,Y)*(N-1)
  SXY <- SXYA
  
  # Se calculan los parametros que permiten identificar el aÃ±o
  # porterior al cambio en la media (T(i)) y la magnitud del cambio (D(i))
  
  
  F <- vector()
  D <- vector()
  T <- vector()
  
  
  for(L in 1:N-1) {
    F[L] <- SX-(((XX[L]-XXM)^2)*N*L)/(N-L)
    D[L] <- ((SX*(YYM-YY[L])- SXY*(XXM-XX[L]))*N)/((N-L)*F[L])
    T[L] <- (L*((N-L)*D[L]^2)*F[L])/(SX*SY-SXY^2)
  }
  
  
  max <- which.max(T)
  maximo <- VAR[max,1]
  resultado<-data.frame(F=F,D=D,T=T)
  
  #testeo
  ndatos<-c(10,15,20,30,40,70,100)
  Ttabla<-c(6.8,7.4,7.8,8.2,8.7,9.3,9.3)
  tabla<-data.frame(ndatos,Ttabla)
  
  if(tabla[2,1]>N & N>tabla[1,1]){
    if(max(T)>tabla[1,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"}
  } else if(tabla[3,1]>N & N>tabla[2,1]){
    if(max(T)>tabla[2,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"}
  } else if(tabla[4,1]>N & N>tabla[3,1]){
    if(max(T)>tabla[3,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"}
  } else if(tabla[5,1]>N & N>tabla[4,1]){
    if(max(T)>tabla[4,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"}
  } else if(tabla[6,1]>N & N>tabla[5,1]){
    if(max(T)>tabla[5,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"}
  } else if(tabla[7,1] >N & N>tabla[6,1]){
    if(max(T)>tabla[6,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"}
  } else if(N>tabla[7,1]){
    if(max(T)>tabla[7,2]){
      m<-"Rechazo Hipotesis nula"
    } else {m<-"no puedo rechazar hipotesis nula"} 
  }  
  
  
  if(tabla[2,1]>N & N>tabla[1,1]){
    if(max(T)>tabla[1,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next}
  } else if(tabla[3,1]>N & N>tabla[2,1]){
    if(max(T)>tabla[2,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next}
  } else if(tabla[4,1]>N & N>tabla[3,1]){
    if(max(T)>tabla[3,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next}
  } else if(tabla[5,1]>N & N>tabla[4,1]){
    if(max(T)>tabla[4,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next}
  } else if(tabla[6,1]>N & N>tabla[5,1]){
    if(max(T)>tabla[5,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next}
  } else if(tabla[7,1] >N & N>tabla[6,1]){
    if(max(T)>tabla[6,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next}
  } else if(N>tabla[7,1]){
    if(max(T)>tabla[7,2]){
      m<-"Rechazo Hipotesis nula"
    } else {next} 
  } else{ 
    m<-"no puedo rechazar hipotesis nula"}
  
  
  if(is.na(T[which(diff(sign(diff(T)))==-2)+1][2])==FALSE){
    x<-readline("Existe un maximo secundario, desea analizarlo?(si,no): ")
    if(x=="si"){
      max2<-T[which(diff(sign(diff(T)))==-2)+1][2]
      maximo2<-VAR[which(T==max2),1]
      posicion2=which(T[]==max2)
      if(tabla[2,1]>N & N>tabla[1,1]){
        if(max2>tabla[1,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[3,1]>N & N>tabla[2,1]){
        if(max2>tabla[2,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[4,1]>N & N>tabla[3,1]){
        if(max2>tabla[3,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[5,1]>N & N>tabla[4,1]){
        if(max2>tabla[4,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[6,1]>N & N>tabla[5,1]){
        if(max2>tabla[5,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[7,1] >N & N>tabla[6,1]){
        if(max2>tabla[6,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"}
      } else if(N>tabla[7,1]){
        if(max2>tabla[7,2]){
          m2<-"Rechazo Hipotesis nula"
        } else {m2<-"no puedo rechazar hipotesis nula"} 
      }  }
    else { 
      m2=NA
    }
  } else {
    m2=NA
  }
  
  if(is.na(T[which(diff(sign(diff(T)))==-2)+1][3])==FALSE){
    x<-readline("Existe un 3er maximo secundario, desea analizarlo?(si,no): ")
    if(x=="si"){
      max3<-T[which(diff(sign(diff(T)))==-2)+1][3]
      maximo3<-VAR[which(T==max3),1]
      posicion3=which(T[]==max3)
      if(tabla[2,1]>N & N>tabla[1,1]){
        if(max3>tabla[1,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[3,1]>N & N>tabla[2,1]){
        if(max3>tabla[2,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[4,1]>N & N>tabla[3,1]){
        if(max3>tabla[3,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[5,1]>N & N>tabla[4,1]){
        if(max3>tabla[4,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[6,1]>N & N>tabla[5,1]){
        if(max3>tabla[5,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[7,1] >N & N>tabla[6,1]){
        if(max3>tabla[6,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"}
      } else if(N>tabla[7,1]){
        if(max3>tabla[7,2]){
          m3<-"Rechazo Hipotesis nula"
        } else {m3<-"no puedo rechazar hipotesis nula"} 
      }  }
    else { 
      m3=NA
    }
  } else {
    m3=NA
  }
  
  if(is.na(T[which(diff(sign(diff(T)))==-2)+1][4])==FALSE){
    x<-readline("Existe un 4to maximo secundario, desea analizarlo?(si,no): ")
    if(x=="si"){
      max4<-T[which(diff(sign(diff(T)))==-2)+1][4]
      maximo4<-VAR[which(T==max4),1]
      posicion4=which(T[]==max4)
      if(tabla[2,1]>N & N>tabla[1,1]){
        if(max4>tabla[1,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[3,1]>N & N>tabla[2,1]){
        if(max4>tabla[2,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[4,1]>N & N>tabla[3,1]){
        if(max4>tabla[3,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[5,1]>N & N>tabla[4,1]){
        if(max4>tabla[4,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[6,1]>N & N>tabla[5,1]){
        if(max4>tabla[5,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"}
      } else if(tabla[7,1] >N & N>tabla[6,1]){
        if(max4>tabla[6,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"}
      } else if(N>tabla[7,1]){
        if(max4>tabla[7,2]){
          m4<-"Rechazo Hipotesis nula"
        } else {m4<-"no puedo rechazar hipotesis nula"} 
      }  }
    else { 
      m4=NA
    }
  } else {
    m4=NA
  }
  
  if(y=="A"){
    if(is.na(m2)==FALSE & is.na(m3)==TRUE & is.na(m4)==TRUE){
      plot(VAR[1:N-1,1] ,T, type="o", col="blue", xlab="Anioo", ylab="T")
      grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
      abline(v =maximo, col="red",untf = FALSE)
      abline(v = VAR[which(T==max2),1],col="red")
      return(list(resultado,Maximo=maximo,Posicion=max,Valor=max(T),m,Maximo2=maximo2,Posicion2=posicion2,Valor2=max2,m2,plot))
    } else if(is.na(m2)==FALSE & is.na(m3)==FALSE & is.na(m4)==TRUE){
      plot(VAR[1:N-1,1] ,T, type="o", col="blue", xlab="Anioo", ylab="T")
      grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
      abline(v =maximo, col="red",untf = FALSE)
      abline(v = VAR[which(T==max2),1],col="red")
      abline(v = VAR[which(T==max3),1],col="red")
      return(list(resultado,Maximo=maximo,Posicion=max,Valor=max(T),m,Maximo2=maximo2,Posicion2=posicion2,Valor2=max2,m2,Maximo3=maximo3,Posicion3=posicion3,Valor3=max3,m3,plot))
    } else if (is.na(m2)==FALSE & is.na(m3)==FALSE & is.na(m4)==FALSE){
      plot(VAR[1:N-1,1] ,T, type="o", col="blue", xlab="Anioo", ylab="T")
      grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
      abline(v =maximo, col="red",untf = FALSE)
      abline(v = VAR[which(T==max2),1],col="red")
      abline(v = VAR[which(T==max3),1],col="red")
      abline(v = VAR[which(T==max4),1],col="red")
      return(list(resultado,Maximo=maximo,Posicion=max,Valor=max(T),m,Maximo2=maximo2,Posicion2=posicion2,Valor2=max2,m2,Maximo3=maximo3,Posicion3=posicion3,Valor3=max3,m3,Maximo4=maximo4,Posicion4=posicion4,Valor4=max4,m4,plot))
    } else if (is.na(m2)==TRUE & is.na(m3)==TRUE & is.na(m4)==TRUE){
      plot(VAR[1:N-1,1] ,T, type="o", col="blue", xlab="Anio", ylab="T")
      grid(nx = NULL, ny = NULL, col = "lightgray", lty = "dotted")
      abline(v =maximo, col="red",untf = FALSE)
      return(list(resultado,Maximo=maximo,Posicion=max,Valor=max(T),m,plot))
    } else {next}
  } else if (y=="D"){
    inicio<-readline("ingrese fecha de inicio del periodo en formato AA/DD/MM:")
    final<-readline("ingrese fecha final del periodo en formato AA/DD/MM:")
    fechas<-seq(as.Date(inicio), as.Date(final), "days")
    datos3=data.frame(fechas[1:N-1], T)
    breaks<-readline("dates_breaks: ")
    plot <- ggplot(datos3, aes(x=fechas[1:N-1], y=T)) +
      geom_line()  +
      xlab("") + 
      theme_bw() +
      scale_x_date(date_breaks = breaks, date_labels = "%d/%b") + 
      theme(axis.text.x=element_text(angle=90, hjust=1))+
      geom_vline(xintercept = as.numeric(fechas[max]),col="red")
    return(list(resultado,Maximo=maximo,Posicion=max,Valor=max(T),m,plot))
  } else {next} 
}
#####ARMONICO#####

#
# Métodos Estadísticos 2016
# María Paula Llano
# DCAO
#
#funcion
#pregunta si guarda el grafico
#

funcion_armonico<-function(datos){
  readline("Recuerde que debe ingresar solo una serie de datos")
  valor<-datos
  N <- length(valor)   
  K<-N/2  #cantidad de armonicos
  P<-N	  
  ###       Calcula el promedio
  PROM <-mean(valor)   #primer termino de la ecuacion
  ## calcula la varianza total
  VAR<-var(valor)
  VAR1<-VAR^0.5
  ## calculo los coeficientes de Fourier   
  CA<-c(1:K)
  CA[]<-0
  PI<-3.1415926
  NARM<-c(1:K) #numero de armonico
  NARM[]<-0
  A<-c(1:K)
  A[]<-0
  B<-c(1:K)
  B[]<-0
  AM<-c(1:K)
  AM[]<-0
  C<-c(1:K)
  C[]<-0
  CA<-c(1:K)
  CA[]<-0
  
  #######comienza
  for (I in 1:(K-1)){
    NARM[I]<-I  
    SUM<-0
    SAM<-0
    
    for (J in 1:N){
      
      SUM<-SUM+valor[J]*sin(I*2*PI*J/P)
      SAM<-SAM+valor[J]*cos(I*2*PI*J/P)
    }
    
    A[I]<-2*SUM/N
    B[I]<-2*SAM/N
    AM[I]<-(A[I]^2+B[I]^2)^0.5
    C[I]<-(((AM[I]^2)/2)/VAR)*100
    CA[I]<-sum(C)
    
    
  }
  
  
  #  calculo del ultimo armonico
  SUM<-0	
  for (J in 1:N){
    SUM<-SUM+(valor[J]*cos(K*2*PI*J/P))
  }
  
  B[K]<-SUM/N
  AM[K]<-B[K]
  C[K]<-((AM[K]^2)/VAR)*100
  CA[K]<- CA[K-1]+C[K]
  A[K]<-0
  NARM[K]<-K
  
  FIN<-matrix(0,K,5, dimnames = list(NARM,c('A', 'B', 'AMPL', 'VAR', 'VAR acu'))) # esta matriz FIN es la que tiene toda la informaci?n
  FIN[,1]<-A
  FIN[,2]<-B
  FIN[,3]<-AM
  FIN[,4]<-round(C,2)
  FIN[,5]<-CA
  print(FIN)
  
  x<-readline("Desea guardar y graficar el periodograma?(si/no): ")
  if(x=="si"){
    nombre<-readline("Nombre con el cual desea guardar el grafico: ")
    png(filename = paste(nombre,".png",sep=""),width = 800,height = 600,units = "px")
    barplot(FIN[,"VAR"],ylim = c(0,100),ylab="Varianza %",xlab = "Armonico",col="green")
    dev.off()
    barplot(FIN[,"VAR"],ylim = c(0,100),ylab="Varianza %",xlab = "Armonico",col="green")
  } else {
    barplot(FIN[,"VAR"],ylim = c(0,100),ylab="Varianza %",xlab = "Armonico",col="green")
  }
  
  return(FIN)
  
}




#####FILONDA####
#
# Métodos Estadísticos 2016
# María Paula Llano
# DCAO
#DESCRIPCION#

# ***** Filtro de la onda del análisis armónico

#funcion#
#pide numero de armonico a filtrar
#pregunta antes de guardar el grafico

#

filonda<-function(valor){
  
  N <- length(valor)
  K<-N/2
  P<-N	
  
  M<-as.numeric(readline("Numero del armonico que desa filtrar: "))
  
  #Calcula el promedio
  
  PROM <-mean(valor)   
  PI<-3.1415926
  
  #coef A y B del armonico a filtrar
  
  SUM<-0
  SAM<-0
  
  for (J in 1:N){
    
    SUM<-SUM+valor[J]*sin(M*2*PI*J/P)
    SAM<-SAM+valor[J]*cos(M*2*PI*J/P)
    
  }
  A<-2*SUM/N
  B<-2*SAM/N
  
  XS<-c(1:N)
  XS[]<-0
  
  FIL<-c(1:N)
  FIL[]<-0
  
  
  for (J in 1:N) {
    XS[J]<-A*sin(2*PI*M*J/P)+B*cos(2*PI*M*J/P)
  }
  
  for (I in 1:N) {
    FIL[I]<- valor[I]-XS[I]
  }
  
  grafico<-readline("Desea guardar el grafico?(si/no): ")
  if(grafico=="si"){
    nombre<-readline("Nombre del grafico a guardar: ")
    png(filename = paste(nombre,".png",sep=""),width = 800,height = 600,units = "px")
    par(mfrow=c(3,1))
    plot(ts(valor))
    plot(ts(XS))
    plot(ts(FIL))
    dev.off()
  }
  return(FIL)
}



#####ESPECTRO#####
#SCRIPT CALCULO ESPECTRO 2018
#DESCRIPCION#
#CALCULA ESPECTOS DE SERIES TEMPORALES DISCRETAS DE ACUERDO AL METODO
# DE BLACKMAN-TUKEY (LIBRO DE OTNES, PAG: 270-272), CON EL CORRESPONDIENTE
#CONTINUO NULO Y BANDAS DE SIGNIFICANCIA AL 5% O AL 10% (CLIMATIC CHANGE,
# NOTA TeCNICA # 79, PAG: 36-42).#
#LA MATRIZ DE ENTRADA DE DATOS DEBE PRESENTAR LAS SERIES A LAS QUE SE LES
#VA A CALCULAR EL ESPECTRO EN COLUMNA. TODAS LAS SERIES DEBEN TENER EL 
# MISMO NUMERO DE REGISTROS (LA MISMA LONGITUD).
# AL COMENZAR EL PROGRAMA REMOVER LAS MEDIAS DE CADA SERIE Y APLICAR UNA
# VENTANA COSENO EN LOS EXTREMOS. LUEGO APLICAR UNA VENTANA HANN, HAMMING
# O PARZEN A ELECCION
# LUEGO GRAFICAR LOS ESPECTROS DE CADA SERIE Y GUARDAR LOS DATOS EN UNA

# MATRIZ CON EL NOMBRE "result" CUYAS COLUMNAS SON:
# col1: frecuencias
# col2: espectro de la serie 1
# col3: limite inferior del intervalo de confianza (serie 1)
# col4: continuo nulo (serie 1)
# col5: limite superior del intervalo de confianza (serie 1)
# col6: espectro de la serie 2
# col7: limite inferior del intervalo de confianza (serie 2)
# col8: continuo nulo (serie 2)
# col9: limite superior del intervalo de confianza (serie 2)
# Y ASI SIGUIENDO CON TODAS LAS SERIES PRESENTES EN LA MATRIZ DE ENTRADA.
# LA MATRIZ "result" PUEDE LUEGO EXPORTARSE PARA GRAFICAR LOS ESPECTROS EN
# ORIGIN Y ALLI TRABAJARLOS.


#\m/LUCIANO ANDRIAN\m/#
#Convertida en funcion.
#lag maximo en funcion del porcentaje del total de datos. 30% por defecto. 
#Pregunta si se quiere modificar

#Pregunta si se desea testear, es necesario testear para graficar y guardar imagen. 
#Ya que no tiene sentido guardar dos graficos de los cuales no se sabe si H0

#El testeo con procedimiento de Anderson, 1 o 2 colas, tiene en cuenta el signo de r1
#imprime por pantalla el H0 y con esto setea que solo realizara el grafico del ruido correspondiente

#Pregunta si se quiere guardar la matriz result que contiene datos de ruido rojo,
##aun no existe matriz para ruido blanco

#Pregunta que resultado devolver con la funcion
##aun no devuelve dos resultados a la vez.
#

espectro<-function(datos){
  
  Matriz<-datos
  p<-readline("Para calular el maximo Lag se usaran el 30% de los datos desea modificar el % (si/no): ")
  if(p=="si"){
    porcentaje<-as.numeric(readline("Porcentaje de datos para el maximo lag: "))
    m<-(porcentaje/100)*(length(Matriz[,1])) 
  } else if(p=="no"){
    m<-0.3*(length(Matriz[,1])) 
  }
  
  ventana<-as.integer(readline(prompt='Ventana: (1)Hann  (2)Hamming  (3)Parzen ? ' ))
  
  intervalo<-as.integer(readline(prompt='Intervalo: (1)95% (2)90% ? '))
  
  n<-dim(Matriz)[1]
  q<-dim(Matriz)[2]
  
  result<-matrix(NA,m+1 ,5)
  f<-rep(NA,m+1)
  
  for (k in 0:m){
    f[k+1]=k/(2*m)
    result[k+1,1]<-f[k+1]
  }
  
  
  # Calcula intervalos de confianza
  
  #print('Grados de libertad')
  gl<-(2*n-(m/2))/m
  
  if (intervalo==1){
    chiinf=(.06414329*gl^1.157371-.09347972)/(1.783669*gl^(-1.319044)+.1701124)
    chisup=(.9295456*gl^.5908365+3.231091)/(.8309965*gl^(-.3875086)-.006014)
  } else {
    chiinf=(.3559888*gl^2.408943-.3692226)/(.6930017*gl^1.299689+4.887728)
    chisup=(3.23178*gl^.9261549+19.64045)/(6.454432*gl^(-.8282872)+1.975337)
  }
  
  #Calculo de la Ventana
  
  u<-rep(NA,m+1)
  if(ventana==1){
    for( r in 0:m){
      u[r+1]<-.5*(1+cos(pi*r/m))
    }
  }
  
  if(ventana==2){
    for (r in 0:m) {
      u[r+1]=.54+.46*cos(pi*r/m)
    }
  }
  
  if(ventana==3){
    for(r in 0:m){
      if (r<=(m/2)){
        u[r+1]=1-6*((r/m)^2)*(1-(r/m))
      }
      if(r>(m/2)){
        u[r+1]=2*(1-(r/m))^3
      }
    }
  }
  
  
  
  for (j in 1:q){
    x<-Matriz[,j]
    
    # Remosi?n de la media de la serie
    
    xm<-mean(x)
    xd<-rep(NA,n)
    for (i in 1:n){
      xd[i]<-x[i]-xm
    }
    
    #Ventana coseno en los extremos
    #l<-floor(.1*n)
    l<-floor(min(.1*n))
    for(i in 1:l){
      c<-.5*(1-cos((i-1)*pi/(l-1)))
      xd[i]<-c*xd[i]
      xd[n-i+1]<-c*xd[n-i+1]
    }
    
    # Funcion de autocovarianzas
    
    Cov<-rep(NA,m+1)
    
    for (r in 0:m){
      s<-0
      for(i in 1:(n-r)){
        s<- s + xd[i]*xd[i+r]
      }
      
      Cov[r+1]<-s/(n-r)
    }
    
    
    # Aplicaci?n de la ventana
    
    R<-Cov*u
    
    #Calculo de espectro y su promedio
    #tengo que definir G
    Gsum<-0
    G<-rep(NA,m+1)
    for (k in 0:m){
      s=0
      for (r in 1:(m-1)){
        s<-s+R[r+1]*cos(pi*r*k/m)
      }
      G[k+1]<-2*(R[1]+2*s+R[m+1]*cos(pi*k))
      Gsum<-Gsum+G[k+1]
    }
    Gmed<-Gsum/(m+1)
    
    
    # Continuo nulo
    #poner corr como matriz
    cor<-cor(x[2:n],x[1:n-1])
    #r1<-cor[1,2]
    r1<-cor
    
    
    #TESTEO 
    test<-readline("Desea testear R1?(Necesario para graficar)(si/no): ")
    if(test=="si"){
      testeo<-as.numeric(readline("Testeo, una(1) o dos(2) colas: "))
      if(testeo==1){
        Rcritico=(-1+1.645*sqrt(length(Matriz[,1])-2))/(length(Matriz[,1])-1)
        if(abs(r1)>abs(Rcritico)){
          print("Ruido Rojo")
          ruido<-"rojo"
        } else {
          print("Ruido Blanco")
          ruido<-"blanco"
        }
      } else if(testeo==2){
        if(r1>0){
          Rcritico=(-1+1.96*sqrt(length(Matriz[,1])-2))/(length(Matriz[,1])-1)
          if(abs(r1)>abs(Rcritico)){
            print("Ruido Rojo")
            ruido<-"rojo"
          } else {
            print("Ruido Blanco")
            ruido<-"blanco"
          }
        } else if(r1<0){
          Rcritico=(-1-1.96*sqrt(length(Matriz[,1])-2))/(length(Matriz[,1])-1)
          if(abs(r1)>abs(Rcritico)){
            print("Ruido Rojo")
            ruido<-"rojo"
          } else {
            print("Ruido Blanco")
            ruido<-"blanco"
          }
        }
      }
    }
    
    
    #ruido rojo
    Cn<-rep(NA,m+1)
    Cnsup<-rep(NA,m+1)
    Cninf<-rep(NA,m+1)
    
    for(k in 0:m){
      Cn[k+1]<-Gmed*((1-r1^2)/(1+r1^2-2*r1*cos(pi*k/m)))
      Cnsup[k+1]<-Cn[k+1]*chisup/gl
      Cninf[k+1]<-Cn[k+1]*chiinf/gl
    }
    
    
    Cnn<-rep(NA,m+1)
    Cnnsup<-rep(NA,m+1)
    Cnninf<-rep(NA,m+1)
    
    #ruido blanco
    for (k in 0:m){
      Cnn[k+1]<-Gmed
      Cnnsup[k+1]<-Cnn[k+1]*chisup/gl
      Cnninf[k+1]<-Cnn[k+1]*chiinf/gl
    }
    
    # Grafica
    
    graficar<-as.numeric(readline("Guardar y graficar(1) o solo graficar(2)(Recuerde que si no testeo no podra realizar esta tarea): "))
    if(graficar==1){
      nombre<-readline("Nombre con el que se guardara la imagen: ")
      if(ruido=="rojo"){
        png(filename=paste(j,nombre,"_rojo",".png",sep=""), width=5500, height=4000, res=600)
        plot(f,G,type='l')
        lines(f,Cninf,col='blue')
        lines(f,Cn,col='red')
        lines(f,Cnsup,col='blue')
        dev.off()
        
        plot(f,G,type='l')
        lines(f,Cninf,col='blue')
        lines(f,Cn,col='red')
        lines(f,Cnsup,col='blue')
        
      } else if(ruido=="blanco"){
        
        png(filename=paste(j,nombre,"_blanco",".png",sep=""), width=5500, height=4000, res=600)
        plot(f,G,type='l')
        lines(f,Cnninf,col='blue')
        lines(f,Cnn,col='red')
        lines(f,Cnnsup,col='blue')
        dev.off()
        
        plot(f,G,type='l')
        lines(f,Cnninf,col='blue')
        lines(f,Cnn,col='red')
        lines(f,Cnnsup,col='blue')
      }
    } else { 
      if(ruido=="rojo"){
        plot(f,G,type='l')
        lines(f,Cninf,col='blue')
        lines(f,Cn,col='red')
        lines(f,Cnsup,col='blue')
        
      } else { plot(f,G,type='l')
        lines(f,Cninf,col='blue')
        lines(f,Cn,col='red')
        lines(f,Cnsup,col='blue')
        
      }
      
    }
    
    for (k in 0:m){
      result[k+1,4*j-2]<-G[k+1]
      result[k+1,4*j-1]<-Cninf[k+1]
      result[k+1,4*j]<-Cn[k+1]
      result[k+1,4*j+1]<-Cnsup[k+1]
    }
    
    # Valores significativos
    
    print('Valores significativos: f, 1/f, Cnsup, G')
    print('Serie n?: ',paste(j))
    
    for (k in 1:m){
      if (G[k+1]>Cnsup[k+1]){
        print(cbind(f[k+1],1/f[k+1],Cnsup[k+1],G[k+1]))
      }
    }
    
  }
  
  colnames(result)<-c('f','G','Cninf','Cn','Cnsup')
  result<-as.data.frame(result)
  
  if(ruido=="rojo"){
    guardar<-readline("Desea guardar la matriz result?(si/no): ")
    if(guardar=="si"){
      nom<-readline("Nombre del archivo .txt: ")
      write.table(result,paste(nom,".txt",sep=""))  
    } 
  } else if(ruido=="blanco"){
    print("no hay matriz para ruido blanco")
  }
  #NO GUARDA resultTADOS DE RUIDO BLANCO...
  
  ##si el espectro sigue un modelo de ruido blanco los resulttados de result no son los correctos
  
  freqsig<-result$f[result$G>=result$Cnsup]
  
  ret<-as.numeric(readline("Return periodos significativos para ruido rojo y r1,Rcritico(1) o matriz reult(2): "))
  if(ret==1){
    return(c(1/freqsig,r1,Rcritico))
  } else if(ret==2){
    return(result)
  }
}