import pandas as pd
import yfinance as yf
import numpy as np

class Helpers():
    @classmethod
    def replace_null(self,df, sym, col, missing):
        ticker = yf.Ticker(sym)

        df.loc[df['Symbol']==sym, col]= ticker.info[missing]
        return  df
    @classmethod
    def remover_outliers(self,nombre_columna, nombre_dataframe,umbral = 1.5):
    # IQR
        Q1 = np.percentile(nombre_dataframe[nombre_columna], 25,
        interpolation = 'midpoint')
        Q3 = np.percentile(nombre_dataframe[nombre_columna], 75,
        interpolation = 'midpoint')
        IQR = Q3 - Q1
        print("Dimensiones viejas: ", nombre_dataframe.shape)
        # Upper bound
        upper = np.where(nombre_dataframe[nombre_columna] >= (Q3+1.5*IQR))
        # Lower bound
        lower = np.where(nombre_dataframe[nombre_columna] <= (Q1-1.5*IQR))
        ''' Removing the Outliers '''
        nombre_dataframe = nombre_dataframe.drop(upper[0])
        nombre_dataframe = nombre_dataframe.drop(lower[0]).reset_index(drop = True)
        print("Nuevas dimensiones: ", nombre_dataframe.shape)
        return nombre_dataframe
    
        
