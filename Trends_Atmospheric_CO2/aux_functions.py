# pip install pandas scipy matplotlib seaborn statsmodels pmdarima
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from scipy.stats import zscore

##################################################################################
# Nota Pessoal: Atualizar as funções utilizando Plotly para gráficos interativos #
##################################################################################

LARGURA = 20 #largura dos gráficos
ALTURA = 10 #altura dos gráficos

def ler_csv(caminho_arquivo:str, average_only:bool=False):
    # Leitura do arquivo adaptado (remoção de comentários)
    dataSet = pd.read_csv(caminho_arquivo)
    
    # Remove coluna 'decimal' se existir
    if 'decimal' in dataSet.columns:
        dataSet = dataSet.drop(columns=['decimal'])
    
    # Formatação da coluna de data
    dataSet['data'] = pd.to_datetime(dataSet['year'].astype(str) + '-' + dataSet['month'].astype(str))
    
    if average_only:
        # Mantém apenas 'data' e 'average'
        dataSet = dataSet.set_index('data')['average']
    else:
        # Remove 'year' e 'month', mantendo todas as outras colunas exceto essas
        colunas = ['data'] + [col for col in dataSet.columns if col not in ['data', 'year', 'month']]
        dataSet = dataSet[colunas]
        
    return dataSet

def plotHistogram(data, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(LARGURA, ALTURA))
    sns.histplot(data, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def plotLine(df, df_label=None, title: str="Gráfico de Linha", xlabel=None, ylabel=None, compare=None, compare_label=None):
    plt.figure(figsize=(LARGURA, ALTURA))
    plt.plot(df, label=df_label, linestyle='-')
    if compare is not None:
        plt.plot(compare, label=compare_label, linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
    
def plotBoxplot(data, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(LARGURA, ALTURA))
    sns.boxplot(x=data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plotScatter(datax, datay, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(LARGURA, ALTURA))
    sns.scatterplot(x=datax, y=datay)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plotArea(data, title=None, xlabel=None, ylabel=None):
    plt.figure(figsize=(LARGURA, ALTURA))
    plt.fill_between(data.index, data.values, alpha=0.5)
    plt.plot(data.index, data.values, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    
    
def plotGraficoQQ(data, title=None):
    plt.figure(figsize=(LARGURA, ALTURA))
    sm.qqplot(data, line='s')
    plt.title(title)
    plt.show()


def plotDecomposicao(data, model='additive', period=12, title=None, figsize=(LARGURA, ALTURA)):

    # Realiza a decomposição
    decomposition = seasonal_decompose(data, model=model, period=period)
    
    # Cria uma figura para plotar os componentes
    plt.figure(figsize=figsize)
    
    # Série Original
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Original')
    plt.legend(loc='upper left')
    plt.title(title)
    
    # Tendência
    plt.subplot(4, 1, 2)
    plt.plot(decomposition.trend, label='Tendência', color='orange')
    plt.legend(loc='upper left')
    
    # Sazonalidade
    plt.subplot(4, 1, 3)
    plt.plot(decomposition.seasonal, label='Sazonalidade', color='green')
    plt.legend(loc='upper left')
    
    # Resíduos
    plt.subplot(4, 1, 4)
    plt.plot(decomposition.resid, label='Resíduos', color='red')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()


def plotAutoCorrelation(data, lags=50):
    plt.figure(figsize=(LARGURA, ALTURA))
    plot_acf(data, ax=plt.gca(), lags=lags)
    plt.title('Função de Autocorrelação (ACF)')
    plt.show()


def plotPartialAutoCorrelation(data, lags=50):
    plt.figure(figsize=(LARGURA, ALTURA))
    plot_pacf(data, ax=plt.gca(), lags=lags)
    plt.title('Função de Autocorrelação Parcial (PACF)')
    plt.show()


def adfullerTest(data):
    adFUllerTest = adfuller(data)
    print(f"Estatística do teste ADF: {adFUllerTest[0]}")
    print(f"Valor-p: {adFUllerTest[1]:.16f}")
    print("Valores críticos:")
    for key, value in adFUllerTest[4].items():
        print(f"   {key}: {value}")

    # Interpretação
    if adFUllerTest[1] <= 0.0005:
        print("A série é estacionária.")
    else:
        print("A série não é estacionária.")


def kolmogorovTest(data):
    ks_test = stats.kstest(data, 'norm')
    
    print('\nTeste de Kolmogorov-Smirnov')
    print(f'Estatística: {ks_test.statistic:.8f}')
    print(f'Valor-p: {ks_test.pvalue:.8f}')
    if ks_test.pvalue < 0.05:
        print('Rejeita a hipótese nula (os dados não seguem uma distribuição normal)')
    else:
        print('Não rejeita a hipótese nula (os dados seguem uma distribuição normal)')


def andersonTest(data):
    ad_test = stats.anderson(data, dist='norm')
    
    print('\nTeste de Anderson-Darling')
    print(f'Estatística: {ad_test.statistic:.8f}')
    print('Valores Críticos:', [f'{cv:.8f}' for cv in ad_test.critical_values])
    print('Níveis de Significância:', ad_test.significance_level)
    
    for i in range(len(ad_test.critical_values)):
        sl, cv = ad_test.significance_level[i], ad_test.critical_values[i]
        if ad_test.statistic > cv:
            print(f'Para um nível de significância de {sl}%, rejeita a hipótese nula (os dados não seguem uma distribuição normal)')
        else:
            print(f'Para um nível de significância de {sl}%, não rejeita a hipótese nula (os dados seguem uma distribuição normal)')
   
            
def detectar_outliers_iqr(series: pd.Series, fator: float = 1.5, true_only: bool = False) -> pd.Series:
    """ 
        Retorna um series com valores booleanos
        Se true_only=True, retorna apenas os valores que são outliers
        Se true_only=False, retorna um series com valores booleanos
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1

    limite_inferior = Q1 - fator * IQR
    limite_superior = Q3 + fator * IQR
    
    outliers = (series < limite_inferior) | (series > limite_superior)
    
    if true_only:
        return series[outliers]
    else:
        return outliers


def detectar_outliers_zscore(series: pd.Series, limite: float = 2.5, true_only: bool = False) -> pd.Series:
    """ 
        Retorna um series com valores booleanos
        Se true_only=True, retorna apenas os valores que são outliers
        Se true_only=False, retorna um series com valores booleanos
    """
    z_scores = zscore(series)
    
    outliers = abs(z_scores) > limite
    if true_only:
        return series[outliers]
    else:  
        return outliers


def auto_arima(data: pd.Series, m: int = 12):
    model = pm.auto_arima(data.dropna(),
                        seasonal=True,
                        m=m,
                        trace=True,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True
                        )
    print(model.summary())