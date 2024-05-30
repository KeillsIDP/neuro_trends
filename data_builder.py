import pandas_datareader.data as web
import pandas as pd
import datetime as dt
import asyncio
import numpy as np
import os 
import glob 
  

with open("TICK.txt") as file: #открываем файл и берем названия акций
    tickers = [row.strip() for row in file]
    
startDate = '2023-03-24' # дата с начала которой будет браться информация о торгах
endDate = dt.datetime.today() # дата до которой будет браться информация
async def getData(ticker):
    try:
        mgWeb = web.DataReader(ticker,'moex',startDate,endDate)[['CLOSE']] # получаем данные о акциях с мосбиржи
        mgWeb.to_excel('./tables/'+ticker+'.xlsx') # сохраняем в таблицу с именем акции
    except Exception as e:
        print(e) # вывод при ошибке
        
async def main():
    for tick in tickers: # для каждой акции запускаем асинхронно функцию получения данных
        await getData(tick)
    tables_files = glob.glob(os.path.join("./tables/", "*.xlsx")) 
    
    data_stock = None
    for file in tables_files:
        data = pd.read_excel(file,index_col="TRADEDATE")
        
        if(data_stock is None):
            data_stock = data
        else:
            data_stock = data_stock.join(data["CLOSE"],how='outer')
        data_stock.rename(columns={'CLOSE':os.path.basename(file).split('.')[0]},inplace=True)
        data_stock.to_excel("./data_stock_test.xlsx")
    
    tables = []
    for file in tables_files:
        data = pd.read_excel(file,index_col="TRADEDATE")
        copy = data.copy()
        if(len(copy.values)==0):
            continue
        copy.values[0] = 0

        close_values = data["CLOSE"]
        for i in range(0,data.shape[0]):
            if(i!=0):
                if(close_values.values[i-1]>close_values.values[i]):
                    copy.values[i] = -1
                elif(close_values.values[i-1]<close_values.values[i]):
                    copy.values[i] = 1
                else:
                    copy.values[i] = 0      
        
        tables.append(copy)                      

    data_stock_labels = tables[0]
    data_stock_labels.rename(columns={'CLOSE':os.path.basename(tables_files[0]).split('.')[0]},inplace=True)

    for i in range(1,len(tables)):
        data_stock_labels = data_stock_labels.join(tables[i]["CLOSE"],how='outer')
        data_stock_labels.rename(columns={'CLOSE':os.path.basename(tables_files[i]).split('.')[0]},inplace=True)
        
    data_stock_labels.fillna(0,inplace=True)
    print(data_stock_labels)
    data_stock_labels.to_excel("./data_stock_labels_test.xlsx")
asyncio.run(main()) # запускаем поток