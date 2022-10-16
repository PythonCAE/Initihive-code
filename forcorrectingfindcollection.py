import pandas as pd
import stumpy
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import traceback


import pymongo
import sys
from io import StringIO
from matplotlib.pyplot import figure


# def update_database(job_id, status, image_url, values):
def update_database(job_id, status, values):
    client = pymongo.MongoClient('mongodb://localhost:27017/')
    db = client.backoffice
    collection = db.forecasting
    mquery = {"uuid": job_id}
    newvalues = {"$set": {"status": status,  "forecast": values}}
    print(mquery, " : ", newvalues)
    collection.update_one(mquery, newvalues)
    # newupdatelist = []
    # for newvalues in values:
    #     newupdatelist.append([newvalues])
    # df = pd.DataFrame(newupdatelist)
    # # plt.table(df)  
    

def  read_data(company):
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["mydatabase"]
        mycol = mydb["Bankcsv"]
        data = mycol.find({"Name":company}).sort("published_date",-1).limit(40)

        list_data = list(data)
        dataframe_data = pd.DataFrame(list_data)
        # print(type(dataframe_data))
        csv_data = dataframe_data.to_csv()
        return csv_data  

def  read_data_query(company): 
        print(company)
        myclient = pymongo.MongoClient("mongodb://localhost:27017/")
        mydb = myclient["mydatabase"]
        mycol = mydb["Bankcsv"]
        data = mycol.find({"Name":company}).limit(20)
        list_data = list(data)
        dataframe_data = pd.DataFrame(list_data)
        csv_data = dataframe_data.to_csv()
        # print(csv_data)
        return csv_data    


def run(query_csv, **ref):
    print(query_csv, ref)
    plt.style.use('https://raw.githubusercontent.com/TDAmeritrade/stumpy/main/docs/stumpy.mplstyle')

    job_id = ref["job_id"]
    # image_url = ref["image_url"]
    start_date = ref["start_date"]
    end_date = ref["end_date"]
    dates = pd.date_range(start_date, end_date)

    r1 = ref["ref1"]
    r2 = ref["ref2"]
    r3 = ref["ref3"]
    


    # ref1_data = pd.read_csv(r1, usecols=['Date', 'Close'], index_col="Date", parse_dates=True, dayfirst=True)
    ref1_datas = read_data(r1)
    ref1_data = StringIO(ref1_datas)
    # print(type(ref1_data))
    df1 = pd.read_csv(ref1_data, usecols=['published_date', 'close'], index_col="published_date", parse_dates=True, dayfirst=True)
    # print(df1_list)
    # df1 = pd.DataFrame(index=dates)
    # df1 = df1.join(readcsv1)
    # df1 = df1.dropna()

    ## print("length of firstlen Query",len(df1['close'].values))


    first_bank_simple_moving_average_5days = round(df1.iloc[:,].rolling(window=5).mean(),2 )
    print("SMA 1st Bank:",first_bank_simple_moving_average_5days.dropna())

    first_bank_simple_moving_average_20days = round(df1.iloc[:,].rolling(window=20).mean(),2 )
    print("SMA 1st Bank:",first_bank_simple_moving_average_20days.dropna())


    # df1['SMA5'] = df1['close'].rolling(window=5,axis=0).mean()
    # df1.dropna(inplace=True)
    # print("hello",df1)
    
    df1_values1 = df1['close'].rolling(window=5,axis=0,center=True).mean()
    df1_values2 = df1['close'].rolling(window=20,axis=0,center=True).mean()
    ax = df1_values1.plot(x="published_date",y="Moving Average")
    df1_values2.plot(ax=ax,x="published_date",y="Moving Average" )
    plt.show()

    # df1.dropna(inplace=True)
    # print("hello",df1)
    

    # values.plot()
    # plt.tight_layout()
    # plt.show()
    # d1 = df1['SMA5']
    # df1['SMA5'].
    # d1.drop('published_date',axis=1)
    # d1.dropna(inplace=True)
    # print(d1['SMA5'])

    # df1.plot(xlabel="published_date",ylabel=['SMA5','SMA20'])



    # print(df1['close'].values)

    if r2 is not None:
        # ref2_data = pd.read_csv(r2, usecols=['Date', 'Close'], index_col="Date", parse_dates=True, dayfirst=True)
        # df2 = pd.DataFrame(index=dates)
        # df2 = df2.join(ref2_data)
        # df2 = df2.dropna()
        ref1_datas = read_data(r2)
        ref1_data = StringIO(ref1_datas)
        # print(type(ref1_data))
        df2 = pd.read_csv(ref1_data, usecols=['published_date', 'close'], index_col="published_date", parse_dates=True, dayfirst=True)
        # print(df1_list)
        # df2 = pd.DataFrame(index=dates)
        # df2 = df2.join(readcsv2)
        # df2 = df2.dropna()

        print("length of firstlen Query",len(df2['close'].values))
        second_bank_simple_moving_average_5days = round(df2.iloc[:5,].rolling(window=5).mean(),2 )
        print("SMA 2nd Bank:",second_bank_simple_moving_average_5days)

        second_bank_simple_moving_average_20days = round(df2.iloc[:20,].rolling(window=20).mean(),2 )
        print("SMA 2nd Bank:",second_bank_simple_moving_average_20days)




    
    if r3 is not None:
        # ref3_data = pd.read_csv(r3, usecols=['Date', 'Close'], index_col="Date", parse_dates=True, dayfirst=True)
        # df3 = pd.DataFrame(index=dates)
        # df3 = df3.join(ref3_data)
        # df3 = df3.dropna()
        ref3_datas = read_data(r3)
        ref3_data = StringIO(ref3_datas)

        df3 = pd.read_csv(ref3_data, usecols=['published_date', 'close'], index_col="published_date", parse_dates=True, dayfirst=True)
        # df3 = pd.DataFrame(index=dates)
        # df3 = df3.join(readcsv3)
        # df3 = df3.dropna()

        third_bank_simple_moving_average_5days = round(df3.iloc[:5,].rolling(window=5).mean(),2 )
        print("SMA 3rd Bank:",third_bank_simple_moving_average_5days)

        third_bank_simple_moving_average_20days = round(df3.iloc[:20,].rolling(window=20).mean(),2 )
        print("SMA 3rd Bank:",third_bank_simple_moving_average_20days)



    # query_csv_path = "/root/go/src/bitbucket.org/sulochan/backoffice/private/forecasting/" + query_csv
    query_datas = read_data_query(query_csv)
    query_data = StringIO(query_datas)

    Q_df = pd.read_csv(query_data)
    Q_z_norm = stumpy.core.z_norm(Q_df['close'].values)
    # print(len(Q_df['close'].values))

    k = 2

    if r1:
        idx_ref1 = stumpy.match(Q_df['close'].values, df1['close'].values )
          
        # print("Bset Matches:",idx_ref1)
        # for iteration in idx_ref1:
        #     print("loop:",iteration)
        idx_ref1 = idx_ref1[0:2]
        # print("idx:",idx_ref1)
        idx1 = np.argmin(idx_ref1)
        # print("After:",idx1)
        # print("index1:",idx1)
        # print(idx)
        # print("Best Matches:",idx)
        # queen_mp = stumpy.stump(T_A = Q_df['close'],
        #                 m =20,
        #                 T_B = df1['close'],
        #                 ignore_trivial = False)
        # print("Hello",queen_mp)                


    if r2:
        idx_ref2 = stumpy.match(Q_df['close'].values, df2['close'].values,max_matches=3)
        idx_ref2 = idx_ref2[0:2]
        # for iteration in idx_ref2:
        #     print(iteration)

    if r3:
        idx_ref3 = stumpy.match(Q_df['close'].values, df3['close'].values,max_matches=3)
        idx_ref3 = idx_ref3[0:2]

    if r3 is not None:
        fig1, axs = plt.subplots(5, 1, constrained_layout=True)
    elif r2 is not None:
        fig1, axs = plt.subplots(4, 1, constrained_layout=True)
    else:
        fig1, axs = plt.subplots(3, 1, constrained_layout=True)

    axs[0].set_title('Pattern Query', fontsize='20')
    axs[0].plot(Q_df['close'].values, lw=2, label="Q_df", color="black")
    axs[0].set_ylabel("Closing Price", fontsize='10')

    axs[1].set_title('Closest matching pattern', fontsize='20')
    # axs[1].plot(df1['close'].values, lw=2, label='r1', color="black")
    axs[1].set_xlabel("Time", fontsize='10')
    axs[1].set_ylabel("Closing Price", fontsize='10')
    axs[1].plot(Q_z_norm, lw=4, label="qtf", color="black")
    for d, i in idx_ref1:
        refx = stumpy.core.z_norm(df1['close'].values[i:i+len(Q_df)+10])
        axs[1].plot(refx, color="blue", lw=2)
    if r2 is not None:
        for d, i in idx_ref2:
            refy = stumpy.core.z_norm(df2['close'].values[i:i+len(Q_df)+10])
            axs[1].plot(refy, color="red", lw=2)
    if r3 is not None:
        for d, i in idx_ref3:
            refz = stumpy.core.z_norm(df3['close'].values[i:i+len(Q_df)+10])
            axs[1].plot(refz, color="green", lw=2)

    if r1 is not None:
        axs[2].set_title(r1, fontsize='20')
        axs[2].plot(df1['close'].values, lw=2, label='r1', color="black")
        axs[2].set_xlabel("Time", fontsize='10')
        axs[2].set_ylabel("Closing Price", fontsize='10')
        axs[2].plot(df1['close'].values, lw=2, label=r1)
 
        stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf

        idx_ref1 = stumpy.match(Q_df['close'].values, df1['close'].values,max_distance=np.inf ,max_matches=3)
        # print("idx_ref1",idx_ref1)
        idx_ref1 = idx_ref1[0:3]
        dataframe =pd.DataFrame(idx_ref1)
        print(dataframe)
        for dis,idx in idx_ref1:
            # for idx in idxs:
               axs[2].plot(range(idx,idx+len(Q_df)),df1.values[idx:idx+len(Q_df)],lw=2)
            #    npt.assert_equal(idx_ref1[:], idx)


        








        
    # stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf

    # matches_top_16 = stumpy.match(
    #     Q_df["close"],
    #     df2["close"],
    #     max_matches=4,      # find the top 16 matches
    # )
    # stumpy.config.STUMPY_EXCL_ZONE_DENOM = 4 # Reset the denominator to its default value
        # # idxs = np.argpartition(distance_profile,topmatch)[:topmatch]
        # idxs = idxs[np.argsort(distance_profile[idxs])]


        # matches_top_three_dataframe=pd.DataFrame(matches_top_three)
        # print(matches_top_three_dataframe)




         
        




        
    #     matches_improved_max_distance = stumpy.match(
    #             Q_df["close"],
    #             df1['close'],
    #             # max_distance=lambda D: max(np.mean(D) - 4 * np.std(D), np.min(D))
    #             max_matches=3
    #     )

    # # Since MASS computes z-normalized Euclidean distances, we should z-normalize our subsequences before plotting
    #     Q_z_norm = stumpy.core.z_norm(Q_df['close'].values)

    #     # axs[2].suptitle('Comparing The Query To All Matches (Improved max_distance)', fontsize='30')
    #     axs[2].set_ylabel('Closes', fontsize='20')

    #     axs[2].set_xlabel('Time', fontsize ='20')
    #     for match_distance, match_idx in matches_improved_max_distance:
    #         match_z_norm = stumpy.core.z_norm(df1['close'].values[match_idx:match_idx+len(Q_df)])
    #         axs[2].plot(match_z_norm, lw=2)
    #     axs[2].plot(Q_z_norm, lw=4, color="black", label="Query Subsequence, Q_df")
    #     # axs[2].legend()
        # axs[2].show()



        
        # distance_profile = stumpy.mass(Q_df['close'],df1['close'])
        # topmatch = 4
        # idxs = np.argpartition(distance_profile,topmatch)[:topmatch]
        # idxs = idxs[np.argsort(distance_profile[idxs])]
        # for idx in idxs:
        #     axs[2].plot(range(idx,idx+len(Q_df)),df1.values[idx:idx+len(Q_df)],lw=2)
      
# To set the exclusion zone to zero, set the denominator to np.`inf`
        

        # for d, i in idx_ref1:
        #     axs[2].plot(range(i, i+len(Q_df)), df1['close'].values[i:i+len(Q_df)], color="blue", lw=4)
       
        # plt.suptitle('Bset Three Nearest Neighborhood ', fontsize='30')
        # plt.xlabel('Close', fontsize ='20')
        # plt.ylabel('Date', fontsize='20')
        # plt.plot(df1)
        # plt.text(0, 4.5, 'Cement', color="black", fontsize=20)
        # plt.text(600, 4.5, 'Cement', color="black", fontsize=20)
        # ax = plt.gca()
        # # rect = Rectangle((500, -4), 3000, 10, facecolor='lightgrey')
        # # ax.add_patch(rect)
        # # plt.text(6000, 4.5, 'Carpet', color="black", fontsize=20)
        
        # plt.show()



        


    if r2 is not None:
        # axs[3].set_title(r2, fontsize='20')
        # axs[3].plot(df2['close'].values, lw=2, label='r2', color="black")

        # axs[2].plot(df2['close'].values, lw=2, label='r2', color="black")
        # axs[3].set_xlabel("Time", fontsize='10')
        # axs[3].set_ylabel("Closing Price", fontsize='10')
        # axs[3].plot(df2['close'].values, lw=2, label=r2)
        # for d, i in idx_ref2:
        #     axs[3].plot(range(i, i+len(Q_df)), df2['close'].values[i:i+len(Q_df)], color="red", lw=4)

        # axs[3].set_title(r2, fontsize='20')
        # axs[3].plot(df2['close'].values, lw=2, label='r1', color="black")
        # axs[3].set_xlabel("Time", fontsize='10')
        # axs[3].set_ylabel("Closing Price", fontsize='10')
        # axs[3].plot(df2['close'].values, lw=2, label=r1)
        # distance_profile = stumpy.mass(Q_df['close'],df2['close'])
        # topmatch = 3
        # idxs = np.argpartition(distance_profile,topmatch)[:topmatch]
        # idxs = idxs[np.argsort(distance_profile[idxs])]
        # print("Best Nearest Neighbouhood Macthes:",idxs)
        # for idx in idxs:
        #     axs[3].plot(range(idx,idx+len(Q_df)),df2.values[idx:idx+len(Q_df)],lw=2)

        axs[3].set_title(r2, fontsize='20')
        axs[3].plot(df2['close'].values, lw=2, label='r1', color="black")
        axs[3].set_xlabel("Time", fontsize='10')
        axs[3].set_ylabel("Closing Price", fontsize='10')
        axs[3].plot(df2['close'].values, lw=2, label=r1)
 
        stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf

        idx_ref2 = stumpy.match(Q_df['close'].values, df2['close'].values,max_distance=np.inf ,max_matches=3)
        idx_ref2 = idx_ref2[0:4]

        dataframe =pd.DataFrame(idx_ref2)
        print(dataframe)
        for dis,idx in idx_ref2:
            axs[3].plot(range(idx,idx+len(Q_df)),df2.values[idx:idx+len(Q_df)],lw=2)



    if r3 is not None:
        # axs[4].set_title(r3, fontsize='20')
        # axs[4].plot(df3['close'].values, lw=2, label='r3', color="black")

        # axs[4].set_xlabel("Time", fontsize='10')
        # axs[4].set_ylabel("Closing Price", fontsize='10')
        # axs[4].plot(df3['close'].values, lw=2, label=r3)
        # for d, i in idx_ref3:
        #     axs[4].plot(range(i, i+len(Q_df)), df3['close'].values[i:i+len(Q_df)], color="green", lw=4)

        # axs[4].set_title(r3, fontsize='20')
        # axs[4].plot(df3['close'].values, lw=2, label='r1', color="black")
        # axs[4].set_xlabel("Time", fontsize='10')
        # axs[4].set_ylabel("Closing Price", fontsize='10')
        # axs[4].plot(df3['close'].values, lw=2, label=r1)
        # distance_profile = stumpy.mass(Q_df['close'],df3['close'])
        # topmatch = 3
        # idxs = np.argpartition(distance_profile,topmatch)[:topmatch]
        # idxs = idxs[np.argsort(distance_profile[idxs])]
        # print("Best Nearest Neighbouhood Macthes:",idxs)

        # for idx in idxs:
        #     axs[4].plot(range(idx,idx+len(Q_df)),df3.values[idx:idx+len(Q_df)],lw=2)
        axs[4].set_title(r3, fontsize='20')
        axs[4].plot(df3['close'].values, lw=2, label='r1', color="black")
        axs[4].set_xlabel("Time", fontsize='10')
        axs[4].set_ylabel("Closing Price", fontsize='10')
        axs[4].plot(df3['close'].values, lw=2, label=r3)
 
        stumpy.config.STUMPY_EXCL_ZONE_DENOM = np.inf

        idx_ref3 = stumpy.match(Q_df['close'].values, df3['close'].values,max_distance=np.inf ,max_matches=3)
        idx_ref3 = idx_ref3[0:3]
        dataframe =pd.DataFrame(idx_ref3)
        print(dataframe)
        for dis,idx in idx_ref3:
            axs[4].plot(range(idx,idx+len(Q_df)),df3.values[idx:idx+len(Q_df)],lw=2)


    # image_save_location = "/root/go/src/bitbucket.org/sulochan/backoffice/" + image_url
    fig1.set_size_inches(20, 12)
    fig1.show()
    


    fig1.savefig('fig6.png')
    # fig1.savefig( dpi=100)

    report = []

    for d, i in idx_ref1:
        r1_dat = {}
        name = "match-" + str(i)
        r1_dat = {"match": name, "price": list(df1['close'].values[i+len(Q_df):i+len(Q_df)+10]), "three_day": 0, "five_day": 0, "ten_day": 0}
        try:
            r1_dat["three_day"] = ((r1_dat["price"][2] - r1_dat["price"][0]) / r1_dat["price"][0]) * 100
        except Exception:
            r1_dat["three_day"] = 0

        try:
            r1_dat["five_day"] = ((r1_dat["price"][4] - r1_dat["price"][0]) / r1_dat["price"][0]) * 100
        except Exception:
            r1_dat["five_day"] = 0

        try:
            r1_dat["ten_day"] = ((r1_dat["price"][9] - r1_dat["price"][0]) / r1_dat["price"][0]) * 100
        except Exception:
            r1_dat["ten_day"] = 0

        a = r1.split("/")
        # b = a[2].split(".")
        b = a[0].split(".")

        r1_display = b[0]
        r1_dat["name"] = r1_display
        report.append(r1_dat)

    if r2:
        for d, i  in idx_ref2:
            r2_dat = {}
            name = "match-" + str(i)
            r2_dat = {"match": name, "price": list(df2['close'].values[i+len(Q_df):i+len(Q_df)+10]), "three_day": 0, "five_day": 0, "ten_day": 0}
            try:
                r2_dat["three_day"] = ((r2_dat["price"][2] - r2_dat["price"][0]) / r2_dat["price"][0]) * 100
            except Exception:
                r2_dat["three_day"] = 0

            try:
                r2_dat["five_day"] = ((r2_dat["price"][4] - r2_dat["price"][0]) / r2_dat["price"][0]) * 100
            except Exception:
                r2_dat["five_day"] = 0

            try:
                r2_dat["ten_day"] = ((r2_dat["price"][9] - r2_dat["price"][0]) / r2_dat["price"][0]) * 100
            except Exception:
                r2_dat["ten_day"] = 0

            a = r2.split("/")
            # b = a[2].split(".")
            b = a[0].split(".")

            r2_display = b[0]
            r2_dat["name"] = r2_display
            report.append(r2_dat)

    if r3:
        for d, i in idx_ref3:
            r3_dat = {}
            name = "match-" + str(i)
            r3_dat = {"match": name, "price": list(df3['close'].values[i+len(Q_df):i+len(Q_df)+10]), "three_day": 0, "five_day": 0, "ten_day": 0}
            try:
                r3_dat["three_day"] = ((r3_dat["price"][2] - r3_dat["price"][0]) / r3_dat["price"][0]) * 100
            except Exception:
                r3_dat["three_day"] = 0

            try:
                r3_dat["five_day"] = ((r3_dat["price"][4] - r3_dat["price"][0]) / r3_dat["price"][0]) * 100
            except Exception:
                r3_dat["five_day"] = 0

            try:
                r3_dat["ten_day"] = ((r3_dat["price"][9] - r3_dat["price"][0]) / r3_dat["price"][0]) * 100
            except Exception:
                r3_dat["ten_day"] = 0

            a = r3.split("/")
            # a = r3.split("/")

            # b = a[2].split(".")
            b = a[0].split(".")

            r3_display = b[0]
            r3_dat["name"] = r3_display
            report.append(r3_dat)

    # for reportiteration in report:
    #     print(reportiteration)        
    fig, ax = plt.subplots()
    dataframe = pd.DataFrame(report ,columns =['match','price','three_day','five_day','ten_day','name'] )
    ax.table(cellText=dataframe.values,colLabels=dataframe.columns,loc='center')
    # fig.tight_layout()
    fig1.set_size_inches(80, 80  )

    # figure(figsize=(80, 40), dpi=80)


    fig.show()
    fig.savefig('fig2.png')


   


    # pdate_database(job_id, "Complete", image_url, report)
    update_database(job_id, "Complete", report)
    # distance_profile = stumpy.mass(Q_df['close'],df1['close'])
    # # print("Close Value",Q_df['close'],df1['close'])
    # idx = np.argmin(distance_profile)
    print("done")



def main():
    
    try:
        job_id = sys.argv[1]
    except Exception:
        print("Cant run without job uuid")
        sys.Exit(1)

    try:
        query = sys.argv[2]
    except Exception:
        print("Cant run without query")
        sys.Exit(1)

    try:
        ref1 = sys.argv[3]
    except Exception:
        print("Cant run without at least 1 ref")
        sys.Exit(1)

    try:
        ref2 = sys.argv[4]
    except Exception:
        ref2 = None

    try:
        ref3 = sys.argv[5]
    except Exception:
        ref3 = None

    # image_url = "/private/forecasting/" + job_id + ".png"
    ref = {"job_id": job_id, "ref1": ref1, "ref2": ref2, "ref3": ref3, "start_date": "2020-01-01", "end_date": "2022-9-21"}

    try:
        run(query, **ref)
    except Exception as e:
        traceback.print_exc()
        print("Error : ", e)
        forecast = []
        # update_database(job_id, "Error", "", forecast)
        update_database(job_id, "Error", forecast)


if __name__ == "__main__":
    main()